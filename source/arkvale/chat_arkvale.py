from typing import Dict, List, Optional, Union

import torch
from opencompass.models.huggingface_above_v4_33 import (
    HuggingFacewithChatTemplate,
    _convert_chat_messages,
    _format_with_fast_chat_template,
    _get_stopping_criteria,
)
from opencompass.registry import MODELS
from transformers import AutoModel, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from arkvale import adapter
import gc

@MODELS.register_module()
class ArkValeChatBot(HuggingFacewithChatTemplate):
    def __init__(
        self,
        path: str,
        page_size: int = 32,
        page_budgets: Optional[Union[int, List[int]]] = 4096 // 32,
        page_topks: Optional[int] = None,
        token_budget: Optional[float] = None,
        n_max_pages: Optional[int] = None,
        n_max_bytes: int = 40 * (1 << 30),
        n_unlimited_layers: int = 2,
        n_max_cpu_pages: Optional[int] = None,
        n_max_cpu_bytes: int = 80 * (1 << 30),
        n_sink_pages: int = 2,
        n_win_pages: int = 2,
        use_sparse_attn: bool = False,
        n_prefetch_layers: Optional[int] = None,
        group_size: Optional[int] = None,
        n_groups: Optional[int] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.path = path
        self.dtype = torch.float16
        self.device: str = "cuda:0"
        
        self.seed = seed

        assert page_topks is not None or token_budget is not None

        self.arkvale_kwargs = dict(
            page_size=page_size,
            page_budgets=page_budgets,
            page_topks=page_topks,
            n_max_pages=n_max_pages,
            n_max_bytes=n_max_bytes,
            n_unlimited_layers=n_unlimited_layers,
            n_max_cpu_pages=n_max_cpu_pages,
            n_max_cpu_bytes=n_max_cpu_bytes,
            n_sink_pages=n_sink_pages,
            n_win_pages=n_win_pages,
            use_sparse_attn=use_sparse_attn,
            n_prefetch_layers=n_prefetch_layers,
            group_size=group_size,
            n_groups=n_groups,
        )

        self.is_static_budget = self.arkvale_kwargs["page_topks"] is not None
        self.token_budget = token_budget

        super().__init__(path, **kwargs)

    def _load_model(self,
                    path: str,
                    kwargs: dict,
                    peft_path: Optional[str] = None,
                    peft_kwargs: Optional[dict] = None):

        hf_logging.disable_progress_bar()
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=self.dtype, device_map=self.device)
        self.model.eval()

        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.do_sample = False

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.is_static_budget:
            adapter.enable_arkvale(
                self.model,
                dtype=self.dtype,
                device=self.device,
                **self.arkvale_kwargs,
            )

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        messages = _convert_chat_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_dict=True,
            add_generation_prompt=True,
            max_length=self.max_seq_len,
        )

        inputs = self.tokenizer.apply_chat_template(messages, **tokenize_kwargs).to(self.model.device)
        in_seq_len = inputs.input_ids.shape[1]

        if not self.is_static_budget:
            page_topks = (int(self.token_budget * in_seq_len) + self.arkvale_kwargs["page_size"] - 1) // self.arkvale_kwargs["page_size"]
            page_topks = max(page_topks, 1)
            self.arkvale_kwargs["page_topks"] = page_topks + self.arkvale_kwargs["n_sink_pages"] + self.arkvale_kwargs["n_win_pages"] - 1
            self.arkvale_kwargs["page_budgets"] = self.arkvale_kwargs["page_topks"] + 1

            torch.cuda.synchronize(self.device)
            old_model = self.model
            self.model = None
            del old_model
            gc.collect()
            torch.cuda.empty_cache()

            self._load_model(self.path, dict())            
            adapter.enable_arkvale(
                self.model,
                dtype=self.dtype,
                device=self.device,
                **self.arkvale_kwargs,
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_out_len, do_sample=False, **kwargs)

        outputs = outputs[:, in_seq_len:]

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decodeds
