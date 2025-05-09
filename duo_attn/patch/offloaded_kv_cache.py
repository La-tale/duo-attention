from transformers.cache_utils import OffloadedCache          #  ← NEW
import torch

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    CausalLMOutputWithPast,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
)

import types
from typing import List, Optional, Tuple, Union

from duo_attn.patch.DuoCache import DuoFull, DuoStreaming


class OffloadedDuoAttentionStaticKVCache:
    def __init__(
        self,
        model,
        full_attention_heads,
        batch_size,
        max_size,
        sink_size,
        recent_size,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.sink_size = sink_size
        self.recent_size = recent_size

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = model.config.hidden_size // self.num_heads

        self.num_full_kv_head_list = [0] * self.num_layers
        self.num_streaming_kv_head_list = [0] * self.num_layers

        self.kv_seq_len_list = [0] * self.num_layers
        self.streaming_kv_seq_len_list = [0] * self.num_layers

        self.streaming_key_states_list = []
        self.streaming_value_states_list = []
        self.full_key_states_list = []
        self.full_value_states_list = []



        self.full_offloadedCache = DuoFull()
        self.streaming_offloadedCache = DuoStreaming()

        for idx, layer_full_attention_heads in enumerate(full_attention_heads):
            layer_full_attention_heads = torch.tensor(layer_full_attention_heads) > 0.5
            num_full_kv_head = layer_full_attention_heads.sum().item()
            num_streaming_kv_head = self.num_kv_heads - num_full_kv_head

            self.num_full_kv_head_list[idx] = num_full_kv_head
            self.num_streaming_kv_head_list[idx] = num_streaming_kv_head

            streaming_key_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            streaming_value_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_key_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_value_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            #  key_states: torch.Tensor,
            #  value_states: torch.Tensor,
            #  layer_idx: int,
            self.full_offloadedCache.update(full_key_states, full_value_states, idx)
            self.streaming_offloadedCache.update(streaming_key_states, streaming_value_states, idx)


    @property
    def streaming_kv_seq_len(self):
        return self.streaming_kv_seq_len_list[-1]

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]
    
    def update_full_kv(self, layer_idx, full_key_states, full_value_states):
        incoming_kv_seq_len = full_key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )
        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        cache_kwargs = {}
        cache_kwargs["kv_seq_len"] = kv_seq_len
        cache_kwargs["incoming_kv_seq_len"] = incoming_kv_seq_len

        key_states, value_states = self.full_offloadedCache.update(full_key_states, full_value_states, layer_idx, cache_kwargs)
        
        ## Should be implemented in update ##
        #  self.full_key_states_list[layer_idx][
        #      :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        #  ].copy_(full_key_states)
        #  self.full_value_states_list[layer_idx][
        #      :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        #  ].copy_(full_value_states)
        #
        return (key_states, value_states)

    def update_streaming_kv(self, layer_idx, streaming_key_states, streaming_value_states):
        ## SMS' COMMENTS ## . 2025-04-30
        ## get_streaming_kv ##
        cache_kwargs = {}
        cache_kwargs["streaming_kv_seq_len"] = self.streaming_kv_seq_len_list[layer_idx]
        ## compress_and_replace_streamingkv ##
        cache_kwargs["sink_size"] = self.sink_size
        cache_kwargs["recent_size"] = self.recent_size
        key_states, value_states = self.streaming_offloadedCache.update(streaming_key_states, streaming_value_states, layer_idx, cache_kwargs)
        
        incoming_kv_seq_len = key_states.shape[1]
        if incoming_kv_seq_len <= self.sink_size + self.recent_size:
            self.streaming_kv_seq_len_list[layer_idx] = incoming_kv_seq_len
        else:
            self.streaming_kv_seq_len_list[layer_idx] = (
                self.recent_size + self.sink_size
            )
        return (key_states, value_states)

    
    def put_full_kv(self, layer_idx, full_key_states, full_value_states):
        incoming_kv_seq_len = full_key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        return self.get_full_kv(layer_idx)

    def compress_and_replace_streaming_kv(
        self, layer_idx, streaming_key_states, streaming_value_states
    ):
        incoming_kv_seq_len = streaming_key_states.shape[1]
        if incoming_kv_seq_len <= self.sink_size + self.recent_size:
            self.streaming_key_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_key_states)
            self.streaming_value_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = incoming_kv_seq_len
        else:
            sink_key_states = streaming_key_states[:, : self.sink_size]
            recent_key_states = streaming_key_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_key_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_key_states
            )
            self.streaming_key_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_key_states)

            sink_value_states = streaming_value_states[:, : self.sink_size]
            recent_value_states = streaming_value_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_value_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_value_states
            )
            self.streaming_value_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = (
                self.recent_size + self.sink_size
            )

    def put(self, layer_idx, key_states, value_states):
        incoming_kv_seq_len = key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )
        if (
            incoming_kv_seq_len + streaming_kv_seq_len
            > self.sink_size + self.recent_size + self.prefilling_chunk_size
        ):
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with sink size {self.sink_size}, recent size {self.recent_size}, and prefilling chunk size {self.prefilling_chunk_size}, current size: {streaming_kv_seq_len}."
            )

        (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        ) = self.split_kv(layer_idx, key_states, value_states)

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.streaming_key_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_key_states)
        self.streaming_value_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_value_states)

        self.update_seq_len(layer_idx, incoming_kv_seq_len)

        return self.get(layer_idx)

    def update_seq_len(self, layer_idx, incoming_kv_seq_len):
        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        self.streaming_kv_seq_len_list[layer_idx] += incoming_kv_seq_len

    def get_full_kv(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
        )

    def get_streaming_kv(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get_unsliced(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            kv_seq_len,
            self.full_key_states_list[layer_idx],
            self.full_value_states_list[layer_idx],
            streaming_kv_seq_len,
            self.streaming_key_states_list[layer_idx],
            self.streaming_value_states_list[layer_idx],
        )

    def split_kv(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        full_key_states = key_states[:, :, :num_full_kv_head, :]
        full_value_states = value_states[:, :, :num_full_kv_head, :]
        streaming_key_states = key_states[:, :, num_full_kv_head:, :]
        streaming_value_states = value_states[:, :, num_full_kv_head:, :]
        return (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        )

    def compress(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if streaming_kv_seq_len <= self.recent_size + self.sink_size:
            return
        recent_key_states = self.streaming_key_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_key_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_key_states)

        recent_value_states = self.streaming_value_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_value_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_value_states)

        self.streaming_kv_seq_len_list[layer_idx] = self.recent_size + self.sink_size

    def clear(self):
        for layer_idx in range(self.num_layers):
            self.kv_seq_len_list[layer_idx] = 0
            self.streaming_kv_seq_len_list[layer_idx] = 0

    def evict_last(self, num_tokens):
        for layer_idx in range(self.num_layers):
            kv_seq_len = self.kv_seq_len_list[layer_idx]
            streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
            self.kv_seq_len_list[layer_idx] = max(0, kv_seq_len - num_tokens)
            self.streaming_kv_seq_len_list[layer_idx] = max(
                0, streaming_kv_seq_len - num_tokens
            )

    @property
    def memory_usage(self):
        memory_usage = 0
        for layer_idx in range(self.num_layers):
            memory_usage += self.full_key_states_list[layer_idx].element_size() * (
                self.full_key_states_list[layer_idx].numel()
            )
            memory_usage += self.full_value_states_list[layer_idx].element_size() * (
                self.full_value_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_key_states_list[layer_idx].element_size() * (
                self.streaming_key_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_value_states_list[
                layer_idx
            ].element_size() * (self.streaming_value_states_list[layer_idx].numel())
        return memory_usage
