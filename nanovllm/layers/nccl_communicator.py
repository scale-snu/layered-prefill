"""
NCCL Communicator for CUDA Graph Compatibility

This module provides a simplified NCCL wrapper for nano-vllm that ensures
CUDA graph compatibility by directly calling NCCL library functions.
"""

import ctypes
import os
import platform
from typing import Optional

import torch
import torch.distributed as dist

# === NCCL Types and Constants ===

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]

class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt64 = 4
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclBfloat16 = 9

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")

class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4

    @classmethod
    def from_torch(cls, op) -> int:
        if op == dist.ReduceOp.SUM:
            return cls.ncclSum
        if op == dist.ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == dist.ReduceOp.MAX:
            return cls.ncclMax
        if op == dist.ReduceOp.MIN:
            return cls.ncclMin
        if op == dist.ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")

def _load_nccl_library():
    """Load NCCL library using ctypes."""
    try:
        # Try to find NCCL library with more comprehensive paths
        if platform.system() == "Linux":
            lib_paths = [
                "libnccl.so",
                "libnccl.so.2",
                "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
                "/usr/local/cuda/lib64/libnccl.so.2",
                "/opt/conda/lib/libnccl.so.2",
                "/usr/local/lib/libnccl.so.2",
                # Add environment variable support
                os.environ.get("NCCL_LIBRARY_PATH", ""),
            ]
        else:
            lib_paths = ["nccl.dll"]
        
        nccl_lib = None
        for lib_path in lib_paths:
            if not lib_path:  # Skip empty paths
                continue
            try:
                nccl_lib = ctypes.CDLL(lib_path)
                break
            except OSError:
                continue
        
        if nccl_lib is None:
            raise RuntimeError("Could not find NCCL library. Please ensure NCCL is installed.")
        
        # Define function signatures
        nccl_lib.ncclGetErrorString.restype = ctypes.c_char_p
        nccl_lib.ncclGetErrorString.argtypes = [ncclResult_t]
        
        nccl_lib.ncclGetUniqueId.restype = ncclResult_t
        nccl_lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        
        nccl_lib.ncclCommInitRank.restype = ncclResult_t
        nccl_lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int
        ]
        
        nccl_lib.ncclAllReduce.restype = ncclResult_t
        nccl_lib.ncclAllReduce.argtypes = [
            buffer_type, buffer_type, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ncclComm_t, cudaStream_t
        ]
        
        return nccl_lib
    except Exception as e:
        raise RuntimeError(f"Failed to load NCCL library: {e}")

class SimpleNCCLCommunicator:
    """Simplified NCCL communicator for nano-vllm."""
    
    def __init__(self):
        self.nccl_lib = _load_nccl_library()
        self.comm = None
        self.initialized = False
    
    def _check_result(self, result: int):
        """Check NCCL result and raise exception if error."""
        if result != 0:
            error_str = self.nccl_lib.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str.decode()}")
    
    def init_comm(self, world_size: int, rank: int):
        """Initialize NCCL communicator."""
        if world_size == 1:
            self.initialized = False
            return
        
        # Get unique ID (rank 0 generates, others receive via broadcast)
        unique_id = ncclUniqueId()
        if rank == 0:
            self._check_result(self.nccl_lib.ncclGetUniqueId(ctypes.byref(unique_id)))
        
        # Broadcast unique ID to all ranks
        unique_id_tensor = torch.ByteTensor(list(unique_id.internal))
        dist.broadcast(unique_id_tensor, src=0)
        for i, byte in enumerate(unique_id_tensor.tolist()):
            unique_id.internal[i] = byte
        
        # Initialize communicator
        comm_ptr = ctypes.c_void_p()
        self._check_result(
            self.nccl_lib.ncclCommInitRank(
                ctypes.byref(comm_ptr), world_size, unique_id, rank
            )
        )
        self.comm = comm_ptr.value
        self.initialized = True
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Perform all-reduce operation using NCCL."""
        if not self.initialized or self.comm is None:
            return tensor
        
        out_tensor = torch.empty_like(tensor)
        current_stream = torch.cuda.current_stream()
        
        self._check_result(
            self.nccl_lib.ncclAllReduce(
                buffer_type(tensor.data_ptr()),
                buffer_type(out_tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(current_stream.cuda_stream)
            )
        )
        
        return out_tensor

# Global NCCL communicator instance
_nccl_comm = None

def _get_nccl_communicator():
    """Get or create NCCL communicator."""
    global _nccl_comm
    if _nccl_comm is None:
        _nccl_comm = SimpleNCCLCommunicator()
        if dist.is_available() and dist.is_initialized():
            _nccl_comm.init_comm(dist.get_world_size(), dist.get_rank())
    return _nccl_comm

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
        try:
            nccl_comm = _get_nccl_communicator()
            return nccl_comm.all_reduce(tensor)
        except Exception as e:
            # Fallback to torch.distributed if NCCL direct call fails
            # This may not be fully CUDA graph compatible, but it's a safe fallback
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor
    return tensor 