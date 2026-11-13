# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Exact two-level FP4 DeOsc over contiguous optimizer master shards.

The model parameter is replicated, but only its shape is used: its values are
stale at the pre-copy optimizer boundary. All quantization statistics come from
the updated master shards after BF16 rounding. One MAX reduction provides the
whole-parameter (whole expert-bank) outer scale. Only tile-row stripes crossed
by shard boundaries are exchanged; interiors remain local. Each exchanged
element has exactly one owner, so SUM is an exact reconstruction, including for
BF16. No zero-filled boundary approximation or complete-weight all-gather is
used. Persistent snapshots/distances and checkpoint tensors are local shards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..fp4_backend import qdq
from ..fp4_contract import TwoLevelFP4Config
from .two_level_fp4_deosc import TwoLevelDeOsc


@dataclass(frozen=True)
class FP4ShardBinding:
    """A model-coordinate interval owned by one rank of the optimizer group.

    ``master`` is a contiguous, flat FP32 tensor for ``[start, end)`` in the
    C-order flattened logical model weight. A non-owner supplies an empty FP32
    tensor and an empty interval. All group ranks supply every logical key.
    """

    model: torch.Tensor
    master: torch.Tensor
    config: TwoLevelFP4Config
    start: int
    end: int


class _ShardQDQPlan:
    def __init__(self, binding, partitions):
        self.binding = binding
        shape = tuple(binding.model.shape)
        self.experts = shape[0] if len(shape) == 3 else 1
        self.rows, self.cols = shape[-2:]
        self.height = 16 if binding.config.weight_2d else 1
        self.expert_size = self.rows * self.cols
        stripes = set()
        if self.expert_size:
            total = self.experts * self.expert_size
            for start, end in partitions:
                if start == end:
                    continue
                for cut in (start, end):
                    if not 0 < cut < total:
                        continue
                    expert, offset = divmod(cut, self.expert_size)
                    if not offset:
                        continue  # An expert boundary is also a scale boundary.
                    row = (offset // self.cols // self.height) * self.height
                    base = expert * self.expert_size + row * self.cols
                    if cut != base:
                        stripes.add((expert, row))
        self.stripes = tuple(sorted(stripes))

    @torch.no_grad()
    def apply(self, process_group, world_size):
        binding = self.binding
        current = binding.master.detach().to(torch.bfloat16).reshape(-1)
        # MAX over BF16 values has the same numerical maximum as the forward
        # FP32 reduction of BF16 inputs. The invalid flag makes NaN rejection
        # collective rather than leaving another rank inside a reduction.
        local_amax = (
            current.abs().amax().float() if current.numel() else current.new_zeros((), dtype=torch.float32)
        )
        summary = torch.stack((local_amax.nan_to_num(), (~torch.isfinite(local_amax)).float()))
        if world_size > 1:
            dist.all_reduce(summary, op=dist.ReduceOp.MAX, group=process_group)
        if bool(summary[1]):
            raise ValueError("Sharded FP4 DeOsc rejects nonfinite BF16-rounded master values")
        outer = (summary[:1] / (6.0 * binding.config.inner_max)).clamp_min(1e-30)

        boundary = current.new_zeros((len(self.stripes), self.height, self.cols))
        for index, (expert, row) in enumerate(self.stripes):
            base = expert * self.expert_size + row * self.cols
            stop = base + min(self.height, self.rows - row) * self.cols
            lo, hi = max(binding.start, base), min(binding.end, stop)
            if hi > lo:
                boundary[index].view(-1)[lo - base : hi - base].copy_(
                    current[lo - binding.start : hi - binding.start]
                )
        if self.stripes and world_size > 1:
            dist.all_reduce(boundary, op=dist.ReduceOp.SUM, group=process_group)

        result = torch.empty_like(current)
        if not current.numel():
            return result
        first_expert = binding.start // self.expert_size
        last_expert = (binding.end - 1) // self.expert_size
        for expert in range(first_expert, last_expert + 1):
            expert_base = expert * self.expert_size
            lo = max(binding.start, expert_base)
            hi = min(binding.end, expert_base + self.expert_size)
            row_start = ((lo - expert_base) // self.cols // self.height) * self.height
            row_end = min(
                self.rows,
                ((hi - expert_base + self.height * self.cols - 1) // (self.height * self.cols)) * self.height,
            )
            slab_base = expert_base + row_start * self.cols
            slab = current.new_zeros((row_end - row_start, self.cols))
            slab.view(-1)[lo - slab_base : hi - slab_base].copy_(
                current[lo - binding.start : hi - binding.start]
            )
            for index, (boundary_expert, row) in enumerate(self.stripes):
                if boundary_expert == expert and row_start <= row < row_end:
                    height = min(self.height, self.rows - row)
                    slab[row - row_start : row - row_start + height].copy_(boundary[index, :height])
            # Per-expert slabs start on logical tile-row boundaries. Original
            # columns are retained, including non-aligned widths; the backend
            # supplies its ordinary trailing padding. The explicit bank-wide
            # outer prevents accidental per-expert/per-shard normalization.
            quantized = qdq(
                slab,
                binding.config,
                binding.config.operand("fwd", "weight"),
                outer_scale=outer,
            )
            result[lo - binding.start : hi - binding.start].copy_(
                quantized.reshape(-1)[lo - slab_base : hi - slab_base]
            )
        return result


class ShardedTwoLevelDeOsc(TwoLevelDeOsc):
    """Same update/window contract as ``TwoLevelDeOsc``, with exact DP QDQ.

    Construction is collective, as is each successful ``step``. Local input
    validation errors, inconsistent active masks and invalid partition coverage
    fail on every participant. Checkpoints deliberately require the same group
    membership and flat partitions: resharding needs an explicit converter.
    """

    def __init__(
        self,
        bindings,
        *,
        process_group=None,
        period=200,
        ratio=4.0,
        start_step=0,
        active_modules=None,
        accumulator_dtype="fp32",
        epsilon=None,
    ):
        initialized = dist.is_available() and dist.is_initialized()
        if process_group is not None and not initialized:
            raise ValueError("A sharded DeOsc process group requires initialized torch.distributed")
        self.process_group = process_group
        self.group_rank = dist.get_rank(process_group) if initialized else 0
        self.world_size = dist.get_world_size(process_group) if initialized else 1
        if self.group_rank < 0 or self.world_size < 1:
            raise ValueError("Current rank is not a member of the DeOsc optimizer group")
        global_rank = dist.get_rank() if initialized else 0
        # Do not throw on just one rank before peers reach metadata exchange.
        local_error, metadata = None, None
        try:
            self.shard_bindings = dict(sorted(dict(bindings).items()))
            if any(not isinstance(item, FP4ShardBinding) for item in self.shard_bindings.values()):
                raise ValueError("Sharded DeOsc bindings must be FP4ShardBinding objects")
            triples = {
                key: (item.model, item.master, item.config) for key, item in self.shard_bindings.items()
            }
            super().__init__(
                triples,
                period=period,
                ratio=ratio,
                start_step=start_step,
                active_modules=active_modules,
                accumulator_dtype=accumulator_dtype,
                epsilon=epsilon,
            )
            metadata = {
                "settings": super()._identity()["settings"],
                "bindings": {
                    key: {
                        "shape": list(item.model.shape),
                        "recipe": item.config.to_dict(),
                        "interval": [item.start, item.end],
                        "model_dtype": str(item.model.dtype),
                        "master_dtype": str(item.master.dtype),
                    }
                    for key, item in self.shard_bindings.items()
                },
            }
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        payload = {"global_rank": global_rank, "error": local_error, "metadata": metadata}
        participants = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(participants, payload, group=self.process_group)
        else:
            participants[0] = payload
        failures = [f"rank {item['global_rank']}: {item['error']}" for item in participants if item["error"]]
        if failures:
            raise ValueError("Sharded DeOsc binding validation failed: " + "; ".join(failures))
        self.group_ranks = [item["global_rank"] for item in participants]
        all_metadata = [item["metadata"] for item in participants]
        self.partitions = self._validate_partition_metadata(all_metadata)
        self._plans = {
            id(item.model): _ShardQDQPlan(item, self.partitions[key])
            for key, item in self.shard_bindings.items()
        }

    def _validate_binding(self, key, model, master, config):
        item = self.shard_bindings[key]
        if not isinstance(config, TwoLevelFP4Config):
            raise ValueError(f"{key}: DeOsc requires an explicit two-level FP4 recipe")
        if not isinstance(model, torch.Tensor) or model.ndim not in (2, 3) or model.dtype != torch.bfloat16:
            raise ValueError(f"{key}: sharded DeOsc requires a 2-D/3-D BF16 model parameter")
        if not isinstance(master, torch.Tensor) or master.dtype != torch.float32 or master.ndim != 1:
            raise ValueError(f"{key}: sharded DeOsc requires a flat FP32 master")
        if not master.is_contiguous() or master.device != model.device:
            raise ValueError(f"{key}: master must be contiguous on the model's device")
        if (
            type(item.start) is not int
            or type(item.end) is not int
            or not 0 <= item.start <= item.end <= model.numel()
        ):
            raise ValueError(f"{key}: invalid master shard interval")
        if item.end - item.start != master.numel():
            raise ValueError(f"{key}: master size differs from its logical shard interval")

    def _validate_partition_metadata(self, all_metadata):
        reference = all_metadata[0]
        for metadata in all_metadata:
            if (
                metadata["settings"] != reference["settings"]
                or metadata["bindings"].keys() != reference["bindings"].keys()
            ):
                raise ValueError("Sharded DeOsc ranks disagree on settings or logical parameter keys")
        partitions = {}
        for key, reference_binding in reference["bindings"].items():
            expected = {name: value for name, value in reference_binding.items() if name != "interval"}
            intervals = []
            for metadata in all_metadata:
                binding = metadata["bindings"][key]
                if {name: value for name, value in binding.items() if name != "interval"} != expected:
                    raise ValueError(f"{key}: sharded DeOsc ranks disagree on shape, dtype or recipe")
                intervals.append(tuple(binding["interval"]))
            cursor = 0
            for start, end in sorted(pair for pair in intervals if pair[0] != pair[1]):
                if start != cursor:
                    raise ValueError(
                        f"{key}: shard partition has a gap or overlap at logical offset {cursor}"
                    )
                cursor = end
            if cursor != self.shard_bindings[key].model.numel():
                raise ValueError(f"{key}: shard partition does not cover the complete model parameter")
            partitions[key] = intervals
        return partitions

    def _master_view(self, model, master):
        return master.reshape(-1)

    def _state_shape(self, key):
        return self.bindings[key][1].shape

    def _weight_qdq(self, master, model, config):
        return self._plans[id(model)].apply(self.process_group, self.world_size)

    def _active_bindings(self):
        flags, errors = [], []
        for key, (_, _, config) in self.bindings.items():
            try:
                flags.append(int(self._active(key, config)))
            except ValueError as error:
                flags.append(-1)
                errors.append(str(error))
        device = next(iter(self.bindings.values()))[1].device
        progress = [self.step_count] + [
            self._states[key]["window_steps"] if key in self._states else -1 for key in self.bindings
        ]
        # A valid local sidecar can still be from a different update on a peer.
        # Detect window/step drift before one rank enters a post-snap collective
        # while another rank proceeds to the next parameter.
        codes = torch.tensor(flags + progress, dtype=torch.int64, device=device)
        extremes = torch.cat((codes, -codes))
        if self.world_size > 1:
            dist.all_reduce(extremes, op=dist.ReduceOp.MAX, group=self.process_group)
        high, low = extremes.chunk(2)
        low = -low
        if bool((low[: len(flags)] < 0).any()):
            detail = "; ".join(errors) if errors else "another rank reported a forward recipe mismatch"
            raise ValueError("Sharded DeOsc active recipe validation failed: " + detail)
        if not torch.equal(low, high):
            raise ValueError("Sharded DeOsc ranks disagree on active routing or observation-window progress")
        return {key: bool(value) for key, value in zip(self.bindings, low[: len(flags)].cpu().tolist())}

    def _identity(self):
        identity = super()._identity()
        identity.update(
            version=3,
            storage_layout="flat-optimizer-shard-v1",
            topology={
                "group_rank": self.group_rank,
                "world_size": self.world_size,
                "group_ranks": self.group_ranks,
            },
        )
        for key, item in self.shard_bindings.items():
            identity["bindings"][key].update(
                shard_interval=[item.start, item.end],
                partitions=[list(pair) for pair in self.partitions[key]],
            )
        return identity

    @property
    def stats(self):
        return {
            **super().stats,
            "local_elements": sum(item.master.numel() for item in self.shard_bindings.values()),
            "partition_world_size": self.world_size,
        }
