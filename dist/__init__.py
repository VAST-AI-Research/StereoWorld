import importlib.util

from .fsdp import shard_model
from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    get_world_group, init_distributed_environment,
                    initialize_model_parallel, sequence_parallel_all_gather,
                    sequence_parallel_chunk, set_multi_gpus_devices,
                    xFuserLongContextAttention)
from .wan_xfuser import usp_attn_forward, usp_attn_s2v_forward, sp_prope_forward

