import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


AC_TOKEN_ID=32001
PAD_TOKEN_ID=32000

# replace modeling_llama.py line 973
@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def llamamodel_forward_noflashattn_fast(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # if self._use_flash_attention_2:
    #     # 2d mask is passed through the layers
    #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    # elif self._use_sdpa and not output_attentions:
    #     # output_attentions=True can not be supported when using SDPA, and we fall back on
    #     # the manual implementation that requires a 4D causal mask in all cases.
    #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
    #         attention_mask,
    #         (batch_size, seq_length),
    #         inputs_embeds,
    #         past_key_values_length,
    #     )
    # else:
    #     # 4d mask is passed through the layers
    #     attention_mask = _prepare_4d_causal_attention_mask(
    #         attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    #     )

    # attn mask by pjh
    global AC_TOKEN_ID
    # attn mask by pjh
    datatype=inputs_embeds.dtype
    datadevice=inputs_embeds.device
    attention_mask = []
    pastpkvi = []

    # assert batch_size == 1
    hasanchor=False
    for i in range(batch_size):
        inputi = input_ids[i]
        # print(inputi)
        mask = torch.full((seq_length, seq_length), torch.finfo(datatype).min, device=datadevice)
        mask_cond = torch.arange(mask.size(-1), device=datadevice)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        # if past_key_values is not None:
        pastpkvi = [i for i in range(past_key_values_length)]
        first=False
        if len(pastpkvi) == 0:
            # print("first")
            first=True

        indextruefalse = inputi==AC_TOKEN_ID[0]
        for idd in AC_TOKEN_ID[1:]:
            indextruefalse += (inputi==idd)
        if torch.sum(indextruefalse) != 0:
            if first:
                hasanchor=True
            acindex = torch.nonzero(indextruefalse).squeeze().tolist()
            start=0
            if not isinstance(acindex, list):
                acindex = [acindex]

            for ix in acindex:
                mask[ix+1:,start:ix] = torch.finfo(datatype).min
                start=ix+1
        
            # for past key value index
            pastpkvi.extend([i+past_key_values_length for i in acindex])
            # pastpkvi.extend(past_key_values_length+acindex)
            maxacindex = acindex[-1]
            pastpkvi.extend([i for i in range(maxacindex+1, seq_length)])

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=datatype, device=datadevice), mask], dim=-1)

        attention_mask.append(mask.unsqueeze(0))

    attention_mask = torch.cat(attention_mask, dim=0).to(inputs_embeds).to(datadevice)
    # attention_mask = mask[None, None, :, :].expand(2, 1, seq_length, seq_length+past_key_values_length)
    attention_mask = attention_mask.unsqueeze(1)



    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        # if use_cache:
        #     next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if use_cache:
            pkvi = layer_outputs[2 if output_attentions else 1]
            if hasanchor:
                next_decoder_cache += ((pkvi[0][:,:,pastpkvi,:], pkvi[1][:,:,pastpkvi,:]),)
                # print(next_decoder_cache)
                # print(idx, next_decoder_cache[0].size(), next_decoder_cache[1].size())
            else:
                # print(pkvi)
                next_decoder_cache += (pkvi,)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_llama_forward(ac_tokenid):
    global AC_TOKEN_ID
    AC_TOKEN_ID = ac_tokenid
    # PAD_TOKEN_ID = pad_tokenid
    transformers.models.llama.modeling_llama.LLamaModel.forward = llamamodel_forward_noflashattn_fast
    