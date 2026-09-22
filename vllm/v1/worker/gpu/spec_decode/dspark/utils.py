# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def _get_dspark_parallel_config(
    parallel_config: ParallelConfig,
    tensor_parallel_size: int,
) -> ParallelConfig:
    if parallel_config.enable_eplb:
        logger.warning_once(
            "EPLB is disabled for the DSpark draft model. EPLB remains enabled "
            "for the target model."
        )

    return replace(
        parallel_config,
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        enable_eplb=False,
        eplb_config=replace(
            parallel_config.eplb_config,
            num_redundant_experts=0,
        ),
        enable_elastic_ep=False,
    )

logger = init_logger(__name__)

# Checkpoint names the token embedding can appear under, most specific first.
_EMBED_KEYS = (
    "embed.weight",
    "model.embed_tokens.weight",
    "model.embed.weight",
    "embed_tokens.weight",
)


def _has_real_weight(module) -> bool:
    """True for a materialised layer; False for PPMissingLayer / None.

    Under pipeline parallelism vLLM replaces the layers a rank does not own with
    PPMissingLayer, which has no ``weight``. Aliasing one of those into the draft
    silently produces a no-op layer rather than an error, so check explicitly.
    """
    return module is not None and getattr(module, "weight", None) is not None


def _load_embed_from_checkpoint(embed: nn.Module, model_path: str) -> None:
    """Fill the draft's own token embedding straight from the checkpoint.

    Only needed under PP: the drafter runs on the LAST pipeline rank, but the
    target's ``embed_tokens`` lives on the FIRST, so there is nothing local to
    alias. Reading the one tensor off disk (~1 GB) avoids adding a cross-rank
    collective to model load, which would have to be ordered against every other
    rank's initialisation.
    """
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    key = shard = None
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for cand in _EMBED_KEYS:
            if cand in weight_map:
                key, shard = cand, os.path.join(model_path, weight_map[cand])
                break
    if key is None:
        raise RuntimeError(
            f"DSpark+PP: could not find a token-embedding tensor in {index_path}; "
            f"looked for {_EMBED_KEYS}. The draft needs its own copy because the "
            "target's embedding lives on PP rank 0."
        )

    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(key)
    with torch.no_grad():
        # VocabParallelEmbedding pads the vocab dimension, so copy into the
        # leading rows rather than assigning the whole tensor.
        embed.weight.data[: w.shape[0]].copy_(
            w.to(dtype=embed.weight.dtype, device=embed.weight.device)
        )
    logger.info(
        "DSpark+PP: loaded draft token embedding %s from key %r", tuple(w.shape), key
    )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
        maybe_share_target_embed,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        parallel_config=_get_dspark_parallel_config(
            vllm_config.parallel_config,
            speculative_config.draft_parallel_config.tensor_parallel_size,
        ),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(vllm_config.load_config),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    if draft_model_config.get_vocab_size() <= target_vocab_size:
        maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head
    elif is_pp and not _has_real_weight(target_lm_head):
        # Should not happen: the drafter runs on the last rank, which owns
        # lm_head. Fail loudly rather than silently drafting through a no-op.
        raise RuntimeError(
            "DSpark+PP: the target lm_head is not materialised on this rank. The "
            "drafter is expected to run on the last pipeline rank."
        )

    return draft_model
