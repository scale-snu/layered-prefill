# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import contextmanager, AbstractContextManager
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from nanovllm import ops

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False


MiB = 1024 * 1024
# Max size for each world size in case symmetric memory is available
# For different SM architectures
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: MiB // 2,  # 512 KB
        8: MiB // 4,  # 256 KB
    },
    "10.0": {
        2: 2 * MiB,  # 2 MB
        4: 2 * MiB,  # 2 MB
        6: 1 * MiB,  # 1 MB
        8: 1 * MiB,  # 1 MB
    }
}

SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: 64 * MiB,  # 64 MB
        8: 64 * MiB,  # 64 MB
    },
    "10.0": {
        2: 8 * MiB,  # 8 MB
        4: 32 * MiB,  # 32 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    }
}


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (inp.storage().nbytes() -
                                   inp.storage_offset() * inp.element_size()
                                   == inp.numel() * inp.element_size())


def is_fully_connected(physical_device_ids: list[int]) -> bool:
    """
    query if the set of gpus are fully connected by nvlink (1 hop)
    """
    import pynvml
    # initialize NVML library
    pynvml.nvmlInit()

    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
    ]
    for i, handle in enumerate(handles):
        for j, peer_handle in enumerate(handles):
            if i < j:
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle,
                        peer_handle,
                        pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                    )
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return False
                except pynvml.NVMLError:
                    print("NVLink detection failed. This is normal if"
                          " your machine has no NVLink equipped.")
                    return False
    return True


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self,
                 max_size=8192 * 1024,
                 symm_mem_enabled=False) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.gloo_group = dist.new_group(backend="gloo")

        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            print("Custom allreduce is disabled because "
                  "of missing custom allreduce library")
            return

        assert dist.get_backend(self.gloo_group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group.")

        rank = dist.get_rank(self.gloo_group)
        self.rank = rank
        world_size = dist.get_world_size(self.gloo_group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            print(
                f"Custom allreduce is disabled due to an unsupported world"
                f" size: {world_size}. Supported world sizes: {str(CustomAllreduce._SUPPORTED_WORLD_SIZES)}. To silence this "
                f"warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        device = torch.device(f"cuda:{dist.get_rank(self.gloo_group)}")

        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = torch.cuda.get_device_capability(device)
        if (torch.cuda.is_available() and symm_mem_enabled
                and device_capability in CUSTOM_ALL_REDUCE_MAX_SIZES):
            max_size = min(
                CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability][world_size],
                max_size)

        device_ids = list(range(torch.cuda.device_count()))
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            device_ids = [
                int(x)
                for x in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
            ]

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id],
                              dtype=torch.int,
                              device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, self.gloo_group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert torch.cuda.is_available()
        fully_connected = is_fully_connected(physical_device_ids)
        if world_size > 2 and not fully_connected:
            print(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly.")
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(ops.meta_size() + max_size,
                                                   uncached=True, group=self.gloo_group)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=self.gloo_group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=self.device)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, rank,
                                       self.fully_connected)
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        # print(f"Registering {len(offset)} cuda graph addresses")
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [[None, None]
                    for _ in range(dist.get_world_size(self.gloo_group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(self.gloo_group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i],
                                       src=rank,
                                       group=self.gloo_group,
                                       device="cpu")
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= self.max_size
        return False

    def all_reduce(self,
                   inp: torch.Tensor,
                   *,
                   out: torch.Tensor = None,
                   registered: bool = False):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank],
                           self.max_size)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(size_in_bytes: int,
                             uncached: Optional[bool] = False,
                             group: Optional[dist.ProcessGroup] = None) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(pointers: list[int],
                           rank: Optional[int] = None,
                           group: Optional[dist.ProcessGroup] = None) -> None:
        if rank is None:
            rank = dist.get_rank(group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])


_ar_comm = None

def _get_ar_communicator():
    """Get or create all-reduce communicator."""
    global _ar_comm
    if _ar_comm is None:
        _ar_comm = CustomAllreduce()
    return _ar_comm

@torch.compiler.disable(recursive=True)
def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Tensor Parallel all-reduce using direct NCCL calls for CUDA graph compatibility.

    This implementation directly calls NCCL library functions to ensure
    CUDA graph compatibility, unlike torch.distributed.all_reduce which
    can cause issues during graph capture.

    Args:
        tensor: Input tensor to be all-reduced

    Returns:
        All-reduced tensor (or original tensor if world_size == 1)
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # try:
        ar_comm = _get_ar_communicator()
        out = ar_comm.custom_all_reduce(tensor)
        if out is not None:
            return out
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor
    return tensor

def capture() -> AbstractContextManager:
    global _ar_comm
    if _ar_comm is None:
        _ar_comm = _get_ar_communicator()
    return _ar_comm.capture() if not _ar_comm.disabled else contextmanager(lambda: (yield))()
