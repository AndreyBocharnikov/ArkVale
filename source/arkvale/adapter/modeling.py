import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from functools import wraps


import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache

# use these classes just for hint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRMSNorm,
)

from arkvale.infer_state import InferState
from arkvale import kernels


def _arkvale_rms_norm_forward(self: LlamaRMSNorm, hidden_states):
    if hidden_states.ndim == 3:
        return kernels.rms_norm(hidden_states, self.weight, self.variance_epsilon)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (hidden_states * self.weight).to(input_dtype)


def _get_mod_n_heads(mod):
    return getattr(mod, "num_heads", mod.config.num_attention_heads)


def _get_mod_n_kv_heads(mod):
    return getattr(mod, "num_key_value_heads", mod.config.num_key_value_heads)


def _get_mod_head_dim(mod):
    return getattr(mod, "head_dim", mod.config.hidden_size // _get_mod_n_heads(mod))


def _get_rope_scale_theta(mod):
    rope_scale = 1.0
    rope_theta = 1e4

    rotary_emb = getattr(mod, "rotary_emb", None)
    if rotary_emb is not None:
        rope_scale = getattr(
            rotary_emb,
            "scaling_factor",
            getattr(rotary_emb, "attention_scaling", rope_scale),
        )
        rope_theta = getattr(rotary_emb, "base", rope_theta)

    rope_params = getattr(mod.config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        rope_scale = rope_params.get("factor", rope_scale)
        rope_theta = rope_params.get("rope_theta", rope_theta)

    rope_scaling = getattr(mod.config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_scale = rope_scaling.get("factor", rope_scale)
        rope_theta = rope_scaling.get("rope_theta", rope_theta)

    return float(rope_scale), float(rope_theta)


def _project_query_states(mod, hidden_states: torch.Tensor):
    bsz, q_len, _ = hidden_states.size()
    n_heads = _get_mod_n_heads(mod)
    head_dim = _get_mod_head_dim(mod)
    query_states = mod.q_proj(hidden_states).view(bsz, q_len, n_heads, head_dim)
    if hasattr(mod, "q_norm"):
        query_states = mod.q_norm(query_states)
    return query_states


def _rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_with_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    q1: Optional[torch.Tensor] = None,
):
    if position_embeddings is None:
        return q, k, q1

    cos, sin = position_embeddings
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(dtype=q.dtype, device=q.device).unsqueeze(2)
    sin = sin.to(dtype=q.dtype, device=q.device).unsqueeze(2)

    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    if q1 is not None:
        q1 = (q1 * cos) + (_rotate_half(q1) * sin)

    return q, k, q1


def _arkvale_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        state.begin_forward(bsz, q_len)

    n_heads = _get_mod_n_heads(self)
    n_kv_heads = _get_mod_n_kv_heads(self)
    head_dim = _get_mod_head_dim(self)
    rope_scale, rope_theta = _get_rope_scale_theta(self)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, n_heads, head_dim)
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)
    value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim)

    kvc = state.kv_caches[cur_id]
    budget = state.layer2budget[cur_id]

    n_pf_layers = state.n_prefetch_layers
    assert n_pf_layers is None or not n_pf_layers

    if position_embeddings is None:
        kernels.qk_apply_rotary_in_place(
            query_states,
            key_states,
            kvc.seq_len,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
    else:
        query_states, key_states, _ = _apply_rotary_with_pos_emb(
            query_states,
            key_states,
            position_embeddings,
        )

    if q_len > 1:
        state.attn_layers[cur_id] = self
        kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)

    state.append_paged_kv_cache(cur_id, key_states, value_states)

    if q_len > 1:
        if budget is not None:
            with torch.cuda.stream(state.prefill_backup_stream):
                state.prefill_backup_pages(cur_id)
                evt = torch.cuda.Event()
                evt.record(state.prefill_backup_stream)
                state.prefill_backup_events[cur_id] = evt
            state.prefill_save_digests(cur_id, key_states)
        attn_output = state.prefill_sdpa(cur_id, query_states)
        infer_state.prefill_evict_extra_pages(
            cur_id, query_states[:, -1:, ...].contiguous()
        )
    else:
        attn_page_ids = kvc.c2p
        if budget is not None and kvc.n_pages > budget:
            _, eids, _ = state.estimate_select_recall(cur_id, query_states)
            assert not state.use_sparse_attn

        attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)

    attn_output_dim = self.o_proj.in_features
    attn_output = attn_output.reshape(bsz, q_len, attn_output_dim)

    attn_output = self.o_proj(attn_output)

    attn_weights = None

    if cur_id == n_layers - 1:
        state.end_forward(bsz, q_len)

    if position_embeddings is None:
        return attn_output, attn_weights, past_key_value
    return attn_output, attn_weights


def enable_arkvale(
    self: LlamaForCausalLM,
    dtype: torch.dtype,
    device: torch.device,
    page_size=32,
    infer_state: InferState = None,
    **kwargs,
):
    if infer_state is None:
        config = self.model.config
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            page_size=page_size,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        self.lm_head.forward = lambda x: _lm_head_forward(x[:, -1:, :])

    if hasattr(self, "config"):
        self.config.use_cache = False
    if hasattr(self, "generation_config"):
        self.generation_config.use_cache = False

    for mod in self.modules():
        mod_cls = str(mod.__class__)
        if "Attention" in mod_cls:
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _arkvale_attn_forward(
                    mod, *args, infer_state=infer_state, **kwargs
                )
            )(mod)
        elif "RMSNorm" in mod_cls:
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _arkvale_rms_norm_forward(
                    mod, *args, **kwargs
                )
            )(mod)

    _old_self_prepare_inputs_for_generation = self.prepare_inputs_for_generation
    _old_self_forward = self.forward

    @wraps(_old_self_prepare_inputs_for_generation)
    def _new_self_prepare_inputs_for_generation(input_ids, *args, **kwargs):
        kwargs["use_cache"] = False
        past_kv = kwargs.get("past_key_values", None)
        if past_kv is not None:
            if isinstance(past_kv, str) and past_kv == "dummy":
                input_ids = input_ids[:, -1:]
                if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                    kwargs["position_ids"] = kwargs["position_ids"][:, -1:]
                elif "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    kwargs["position_ids"] = (
                        kwargs["attention_mask"].long().sum(dim=-1, keepdim=True) - 1
                    )
                elif "cache_position" in kwargs and kwargs["cache_position"] is not None:
                    cache_pos = kwargs["cache_position"]
                    if cache_pos.ndim == 1:
                        kwargs["position_ids"] = cache_pos[-1:].view(1, 1).expand(
                            input_ids.shape[0], 1
                        )
                    else:
                        kwargs["position_ids"] = cache_pos[:, -1:]
            kwargs["past_key_values"] = None
        return _old_self_prepare_inputs_for_generation(input_ids, *args, **kwargs)

    @wraps(_old_self_forward)
    def _new_self_forward(*args, **kwargs):
        kwargs["use_cache"] = False
        ret = _old_self_forward(*args, **kwargs)
        if isinstance(ret, dict):
            ret["past_key_values"] = "dummy"
        elif hasattr(ret, "past_key_values"):
            ret.past_key_values = "dummy"
        return ret

    self.prepare_inputs_for_generation = _new_self_prepare_inputs_for_generation
    self.forward = _new_self_forward

    return self
