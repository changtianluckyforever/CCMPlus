from typing import Dict

import torch
import torch.distributed as dist
import sys

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None  # type: ignore


def is_distributed() -> bool:
    """
    Checks if the distributed process group is available and has been initialized
    """
    return dist.is_available() and dist.is_initialized()


def peak_cpu_memory() -> Dict[int, int]:
    """
    Get peak memory usage for each worker, as measured by max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, otherwise the result will be 0.0 for every worker.
    """
    if resource is None or sys.platform not in ("linux", "darwin"):
        peak_bytes = 0
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # On OSX the result is in bytes.
            peak_bytes = peak
        else:
            # On Linux the result is in kilobytes.
            peak_bytes = peak * 1_024

    if is_distributed():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes])
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0]) for _ in range(world_size)]

        # If the backend is 'nccl', this means we're training on GPUs, so these tensors
        # need to be on GPU.
        if dist.get_backend() == "nccl":
            peak_bytes_tensor = peak_bytes_tensor.cuda()
            gather_results = [x.cuda() for x in gather_results]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict: Dict[int, int] = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

        return results_dict
    else:
        return {0: peak_bytes}


def peak_gpu_memory() -> Dict[int, int]:
    """
    Get the peak GPU memory usage in bytes by device.
    # Returns
    `Dict[int, int]`
        Keys are device ids as integers.
        Values are memory usage as integers in bytes.
        Returns an empty `dict` if GPUs are not available.
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()

    results_dict: Dict[int, int] = {}
    if is_distributed():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_bytes_tensor)

        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])
    else:
        results_dict = {0: torch.cuda.max_memory_allocated()}

    # Reset peak stats.
    torch.cuda.reset_max_memory_allocated(device)

    return results_dict
