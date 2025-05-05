import torch

import triton
import triton.language as tl

import numpy as np
from numba import cuda

from cupyx import jit
import cupy

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@jit.rawkernel()
def elementwise_add_kernel(x, y, out, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        out[i] = x[i] + y[i]

def add_cupy(x, y): 
    n_elements = x.size()[0]
    out_tensor = torch.empty_like(x)

    cupy_x = cupy.asarray(x)
    cupy_y = cupy.asarray(y)
    cupy_out = cupy.asarray(out_tensor)
    
    block_size = 256
    grid_size = (n_elements + block_size - 1) // block_size
    elementwise_add_kernel((grid_size,), (block_size,), (cupy_x, cupy_y, cupy_out, n_elements))
    return out_tensor

@triton.jit
def add_kernel_triton(x_ptr,
               y_ptr, 
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel_triton[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

@cuda.jit
def add_kernel_numba(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    c[tid] = a[tid] + b[tid]

def add_numba(x: torch.Tensor, y: torch.Tensor):
    n = x.numel()
    out_tensor = torch.empty_like(x)

    numba_x = cuda.as_cuda_array(x)
    numba_y = cuda.as_cuda_array(y)
    numba_out = cuda.as_cuda_array(out_tensor)

    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    add_kernel_numba[blocks_per_grid, threads_per_block](
        numba_x, numba_y, numba_out 
    )
    return out_tensor

if __name__ == '__main__':
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add_triton(x, y)
    output_numba = add_numba(x, y)
    output_cupy = add_cupy(x.cpu(), y.cpu()).cuda()

    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    print(f'The maximum difference between torch and numba is '
        f'{torch.max(torch.abs(output_torch - output_numba))}')
    print(f'The maximum difference between torch and cupy is '
        f'{torch.max(torch.abs(output_torch - output_cupy))}')