# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident suffix decoding proposer.

Wraps the SuffixGPU library (device-tensor drafter with a local
per-request matcher and a cross-request global suffix index) behind the
same device-state contract as NgramProposerGPU: the previous step's
sampled ids are scattered into a resident token buffer and drafting
runs entirely on device, so the proposer composes with async
scheduling (no host sync on the draft path).

The draft path runs eagerly (fused Triton kernels) by default at first
call; when ``suffix_gpu_use_cuda_graph`` is enabled the whole
update+propose chain is captured once into a CUDA graph on fixed
max-batch buffers and replayed each step, falling back to eager if
capture fails.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


class SuffixProposerGPU:
    """Device-state suffix-decoding drafter (async-scheduling safe)."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 runner=None):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"

        # Lazy import so vLLM works without the SuffixGPU package.
        from suffix_gpu.proposer import SuffixGPUDrafter

        self.k = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_cuda_graph = config.suffix_gpu_use_cuda_graph
        self.ingest_chunk = config.suffix_gpu_ingest_chunk
        enable_global = config.suffix_decoding_max_cached_requests != 0

        self.drafter = SuffixGPUDrafter(
            k=self.k,
            device=device,
            max_pattern_len=config.suffix_decoding_max_tree_depth,
            min_match_len=1,
            max_occurrences=config.suffix_gpu_max_occurrences,
            enable_global=enable_global,
            global_capacity=config.suffix_gpu_global_capacity,
            delta_capacity=config.suffix_gpu_delta_capacity,
            rebuild_stream=torch.cuda.Stream(device)
            if device.type == "cuda" else None,
            max_spec_factor=config.suffix_decoding_max_spec_factor,
            max_spec_offset=0.0,
            min_token_prob=config.suffix_decoding_min_token_prob,
        )

        # CUDA-graph state (captured lazily on the first propose call,
        # after eager warmup has JIT-compiled the Triton kernels).
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_failed = False
        self._g_num_tokens: torch.Tensor | None = None
        self._g_sampled: torch.Tensor | None = None
        self._g_counts: torch.Tensor | None = None
        self._g_token_ids: torch.Tensor | None = None
        self._g_draft: torch.Tensor | None = None
        self._g_nv: torch.Tensor | None = None

        self._warmed_up = False

    # Reuse NgramProposerGPU's device-side bookkeeping helper verbatim
    # (it only depends on self.device).
    update_token_ids_ngram = NgramProposerGPU.update_token_ids_ngram

    def _warmup(self, token_ids_gpu: torch.Tensor) -> None:
        """JIT-compile the Triton kernels on dummy max-shape data."""
        b = self.max_num_seqs
        s = token_ids_gpu.shape[1]
        buf = torch.zeros(b, s, dtype=torch.int32, device=self.device)
        counts = torch.randint(1, max(2, min(64, s // 2)), (b,),
                               dtype=torch.int32, device=self.device)
        sampled = torch.full((b, self.k + 1), -1, dtype=torch.int32,
                             device=self.device)
        sampled[:, 0] = 1
        for _ in range(3):
            self.drafter.propose_with_update(
                counts, buf, sampled, max_model_len=self.max_model_len)
        torch.cuda.synchronize(self.device)
        self._warmed_up = True

    def _capture(self, token_ids_gpu: torch.Tensor,
                 sampled_width: int) -> None:
        """Capture update+propose on fixed max-batch buffers.

        token_ids_gpu is the runner's persistent buffer, so the graph
        binds its storage directly; per-step inputs are staged into
        fixed buffers before replay.
        """
        b = self.max_num_seqs
        self._g_num_tokens = torch.zeros(b, dtype=torch.int32,
                                         device=self.device)
        self._g_sampled = torch.full((b, sampled_width), -1,
                                     dtype=torch.int32, device=self.device)
        self._g_counts = torch.zeros(b, dtype=torch.int64,
                                     device=self.device)
        self._g_token_ids = token_ids_gpu
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._g_draft, self._g_nv, _ = self.drafter.propose_with_update(
                self._g_num_tokens, self._g_token_ids, self._g_sampled,
                self._g_counts, max_model_len=self.max_model_len)
        self._graph = graph
        logger.info_once("suffix_gpu: draft path captured into CUDA graph "
                         "(max_batch=%d, sampled_width=%d)", b,
                         sampled_width)

    def propose(
        self,
        num_speculative_tokens: int,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        valid_sampled_token_ids_gpu: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft tokens for the batch; mirrors NgramProposerGPU.propose.

        Scatters the sampled ids into token_ids_gpu, then matches and
        drafts on device. Returns (draft_tokens [B, k] int32,
        num_valid_draft_tokens [B] int32).
        """
        assert num_speculative_tokens == self.k
        assert token_ids_gpu.device == self.device

        # Host-side upkeep: swap in finished background SA rebuilds.
        self.drafter.poll()

        if not self._warmed_up:
            self._warmup(token_ids_gpu)

        bs = num_tokens_no_spec.shape[0]
        width = valid_sampled_token_ids_gpu.shape[1]

        use_graph = (self.use_cuda_graph and not self._graph_failed
                     and token_ids_gpu.shape[0] == self.max_num_seqs)
        if use_graph and self._graph is None:
            try:
                self._capture(token_ids_gpu, width)
            except Exception:
                logger.exception(
                    "suffix_gpu: CUDA graph capture failed; "
                    "falling back to eager kernels.")
                self._graph_failed = True
                self._graph = None
        if (self._graph is not None and use_graph
                and self._g_token_ids is token_ids_gpu
                and self._g_sampled.shape[1] == width):
            self._g_num_tokens[:bs].copy_(num_tokens_no_spec)
            self._g_num_tokens[bs:].zero_()
            self._g_sampled[:bs].copy_(valid_sampled_token_ids_gpu)
            self._g_sampled[bs:].fill_(-1)
            self._g_counts[:bs].copy_(valid_sampled_tokens_count)
            self._g_counts[bs:].zero_()
            self._graph.replay()
            return self._g_draft[:bs], self._g_nv[:bs]

        draft, num_valid, _ = self.drafter.propose_with_update(
            num_tokens_no_spec, token_ids_gpu,
            valid_sampled_token_ids_gpu, valid_sampled_tokens_count,
            max_model_len=self.max_model_len)
        return draft, num_valid

    # ------------------------------------------------------------------
    # global-memory ingestion (host-side, off the draft path)
    # ------------------------------------------------------------------
    def ingest_active_requests(self, input_batch: InputBatch,
                               token_ids_gpu: torch.Tensor) -> None:
        """Chunked incremental ingestion of in-flight responses."""
        if self.drafter.global_index is None:
            return
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        num_tokens = input_batch.num_tokens_no_spec
        num_prompt = input_batch.num_prompt_tokens
        for req_id, idx in input_batch.req_id_to_index.items():
            resp_len = int(num_tokens[idx]) - int(num_prompt[idx])
            if resp_len < self.ingest_chunk:
                continue
            start = int(num_prompt[idx])
            keys.append(req_id)
            rows.append(token_ids_gpu[idx, start:start + resp_len])
            lengths.append(resp_len)
        if keys:
            self.drafter.ingest_active(keys, rows, lengths,
                                       chunk=self.ingest_chunk)

    def on_requests_finished(self, finished_req_ids, input_batch: InputBatch,
                             token_ids_gpu: torch.Tensor) -> None:
        """Final-flush finished requests before their rows are reused."""
        if self.drafter.global_index is None:
            return
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        num_tokens = input_batch.num_tokens_no_spec
        num_prompt = input_batch.num_prompt_tokens
        for req_id in finished_req_ids:
            idx = input_batch.req_id_to_index.get(req_id)
            if idx is None:
                self.drafter._ingested.pop(req_id, None)
                continue
            resp_len = int(num_tokens[idx]) - int(num_prompt[idx])
            start = int(num_prompt[idx])
            keys.append(req_id)
            rows.append(token_ids_gpu[idx, start:start + max(resp_len, 0)])
            lengths.append(max(resp_len, 0))
        if keys:
            self.drafter.ingest_active(keys, rows, lengths, final=True,
                                       chunk=self.ingest_chunk)

    def load_model(self, *args, **kwargs) -> None:
        pass

    def dummy_run(self, num_tokens: int = 1) -> None:
        pass
