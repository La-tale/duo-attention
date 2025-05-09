import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from transformers.cache_utils import OffloadedCache, DynamicCache

class DuoFull(OffloadedCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    ## SMS' COMMENTS ## . 2025-04-30
    ## Same ##
    #  def prefetch_layer(self, layer_idx: int):
    #      "Starts prefetching the next layer cache"
    #      if layer_idx < len(self):
    #          with torch.cuda.stream(self.prefetch_stream):
    #              # Prefetch next layer tensors to GPU
    #              device = self.original_device[layer_idx]
    #              self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
    #              self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)
    #
    #  def evict_previous_layer(self, layer_idx: int):
    #      "Moves the previous layer cache to the CPU"
    #      if len(self) > 2:
    #          # We do it on the default stream so it occurs after all earlier computations on these tensors are done
    #          prev_layer_idx = (layer_idx - 1) % len(self)
    #          self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
    #          self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            ## SMS' COMMENTS ## . 2025-04-30
            kv_seq_len = cache_kwargs["kv_seq_len"]
            incoming_kv_seq_len = cache_kwargs["incoming_kv_seq_len"]

            key_tensor[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(key_states)
            value_tensor[
                :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
            ].copy_(value_states)

            self.key_cache[layer_idx] = key_tensor
            self.value_cache[layer_idx] = value_tensor
            #
            #  self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            #  self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None

class DuoStreaming(OffloadedCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    ## SMS' COMMENTS ## . 2025-04-30
    ## Same ##
    #  def prefetch_layer(self, layer_idx: int):
    #      "Starts prefetching the next layer cache"
    #      if layer_idx < len(self):
    #          with torch.cuda.stream(self.prefetch_stream):
    #              # Prefetch next layer tensors to GPU
    #              device = self.original_device[layer_idx]
    #              self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
    #              self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)
    #
    #  def evict_previous_layer(self, layer_idx: int):
    #      "Moves the previous layer cache to the CPU"
    #      if len(self) > 2:
    #          # We do it on the default stream so it occurs after all earlier computations on these tensors are done
    #          prev_layer_idx = (layer_idx - 1) % len(self)
    #          self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
    #          self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            ## SMS' COMMENTS ## . 2025-04-30
            ## Forward in llama.py ##
            ## Index from get_streaming_kv ##
            streaming_kv_seq_len = cache_kwargs["streaming_kv_seq_len"]
            streaming_key_states = torch.cat(
                [key_tensor[:, :streaming_kv_seq_len], key_states], dim=1
            )
            streaming_value_states = torch.cat(
                [value_tensor[:, :streaming_kv_seq_len], value_states], dim=1
            )
            
            ## compress_and_replace_streaming_kv ##
            sink_size = cache_kwargs["sink_size"]
            recent_size = cache_kwargs["recent_size"]
            incoming_kv_seq_len = streaming_key_states.shape[1]
            
            if incoming_kv_seq_len <= sink_size + recent_size:
                key_tensor[
                    :,
                    :incoming_kv_seq_len,
                ].copy_(streaming_key_states)
                value_tensor[
                    :,
                    :incoming_kv_seq_len,
                ].copy_(streaming_value_states)
            else:
                sink_key_states = streaming_key_states[:, : sink_size]
                recent_key_states = streaming_key_states[
                    :, incoming_kv_seq_len - recent_size : incoming_kv_seq_len
                ]
                key_tensor[:, : sink_size].copy_(
                    sink_key_states
                )
                key_tensor[
                    :, sink_size : sink_size + recent_size
                ].copy_(recent_key_states)

                sink_value_states = streaming_value_states[:, : sink_size]
                recent_value_states = streaming_value_states[
                    :, incoming_kv_seq_len - recent_size : incoming_kv_seq_len
                ]
                value_tensor[:, : sink_size].copy_(
                    sink_value_states
                )
                value_tensor[
                    :, sink_size : sink_size + recent_size
                ].copy_(recent_value_states)


            self.key_cache[layer_idx] = key_tensor
            self.value_cache[layer_idx] = value_tensor

            return streaming_key_states, streaming_value_states 
            #
            #  self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            #  self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None

