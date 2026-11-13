# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Versioned FP4 SR implementations and private counter allocation.

The GPU implementation is Philox4x32-10 with a 64-bit block counter in
counter words 0/1, zero counter words 2/3, and the reservation's 64-bit seed
as its key. Four consecutive logical padded elements consume the four output
words of one block. A word's high 24 bits times 2**-24 gives a uniform in
[0, 1). Kernel tiling does not change this mapping. RHT continues to use the
small, private CPU reference generator in both modes.
"""

REFERENCE_RNG_VERSION = "torch-cpu-reservation-sha256-v1"
PHILOX_SR_RNG_VERSION = "philox4x32-10-counter64-u24-v1"


def reserve_philox_sr(random, words):
    """Reserve blocks without allocating a random tensor or touching global RNG."""
    if type(words) is not int or words < 0:
        raise ValueError("Philox SR requires a nonnegative integer word count")
    blocks = (words + 3) // 4
    if random.sr_offset + blocks > 2**64:
        raise ValueError("Philox SR exhausted its 64-bit block counter")
    return random.reserve("sr", blocks)


def rng_implementation(configs):
    """Keep the old checkpoint identity unchanged when all SR is CPU based."""
    implementations = {config.sr_rng for config in configs}
    if implementations <= {"torch_cpu"}:
        return REFERENCE_RNG_VERSION
    return {
        "sr": sorted(
            REFERENCE_RNG_VERSION if mode == "torch_cpu" else PHILOX_SR_RNG_VERSION
            for mode in implementations
        ),
        "rht": REFERENCE_RNG_VERSION,
    }
