"""GPU utilities for multi-worker evaluation."""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_worker_gpus(worker_id, gpu_ids=None, gpus_per_worker=1):
    """Set CUDA_VISIBLE_DEVICES for a worker and return (device, llm_device_map)."""
    if gpu_ids is None:
        gpu_ids = list(range(4))

    start_idx = worker_id * gpus_per_worker
    if start_idx + gpus_per_worker > len(gpu_ids):
        raise ValueError(
            f"Worker {worker_id} needs GPUs [{start_idx}:{start_idx + gpus_per_worker}] "
            f"but only {len(gpu_ids)} GPUs available."
        )

    worker_gpus = gpu_ids[start_idx : start_idx + gpus_per_worker]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in worker_gpus)

    device = "cuda"
    llm_device_map = "auto" if gpus_per_worker > 1 else device

    logger.info(
        f"Worker {worker_id}: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, "
        f"llm_device_map={llm_device_map}"
    )
    return device, llm_device_map


def validate_gpu_config(num_workers, gpu_ids, gpus_per_worker=1):
    """Validate that num_workers * gpus_per_worker <= len(gpu_ids)."""
    if gpu_ids is None:
        gpu_ids = list(range(4))
    required = num_workers * gpus_per_worker
    if required > len(gpu_ids):
        raise ValueError(
            f"Need {required} GPUs ({num_workers} workers x {gpus_per_worker}/worker) "
            f"but only {len(gpu_ids)} available."
        )
