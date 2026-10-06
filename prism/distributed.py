"""Optional native SP and BF16 FSDP2 for a torchrun worker, outside the UI process."""
import torch


def initialize(sp_size):
    if type(sp_size) is not int or sp_size < 1:
        raise ValueError("sp_size must be a positive integer")
    import torch.distributed as dist
    from .native.utils.parallel_states import initialize_parallel_state
    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        size = dist.get_world_size()
        if size % sp_size:
            raise ValueError("world_size must be divisible by sp_size")
        state = initialize_parallel_state(sp=sp_size, dp_replicate=size // sp_size)
        return torch.device("cuda", local_rank), state
    except BaseException:
        dist.destroy_process_group()
        raise


def shard(bridge, mesh):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    # Keep explicit FP32 timestep parameters and BF16 block parameters as stored.
    # ConvRot weights are buffers and must not be mistaken for FSDP-sharded params.
    policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None)
    for block in bridge.get_fsdp_block_list():
        fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(bridge, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    return bridge
