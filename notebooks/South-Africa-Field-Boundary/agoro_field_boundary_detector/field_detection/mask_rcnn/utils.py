"""Utilisation classes and methods, code from https://github.com/pytorch/vision."""
import datetime
import errno
import os
import pickle  # noqa S403
import time
from collections import defaultdict, deque
from typing import Any, Optional

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> None:
        """Initialise the smoothed value."""
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # type: ignore
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: Any, n: int = 1) -> None:
        """Update the smoothed value with the given value."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Warning: does not synchronize the deque."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")  # type: ignore
        dist.barrier()  # type: ignore
        dist.all_reduce(t)  # type: ignore
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> Any:
        """Get the median value."""
        d = torch.tensor(list(self.deque))  # type: ignore
        return d.median().item()

    @property
    def avg(self) -> Any:
        """Get the average value."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)  # type: ignore
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        """Get the global average."""
        return self.total / self.count

    @property
    def max(self) -> float:
        """Get the maximum value."""
        return max(self.deque)  # type: ignore

    @property
    def value(self) -> float:
        """Get the value."""
        return self.deque[-1]  # type: ignore

    def __str__(self) -> str:
        """Smoothed value representation."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data: Any) -> Any:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    :param data: any picklable object.
    :return: list[data]: list of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")  # type: ignore
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]  # type: ignore
    dist.all_gather(size_list, local_size)  # type: ignore
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))  # type: ignore
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")  # type: ignore
        tensor = torch.cat((tensor, padding), dim=0)  # type: ignore
    dist.all_gather(tensor_list, tensor)  # type: ignore

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))  # noqa S301
    return data_list


def reduce_dict(input_dict: Any, average: bool = True) -> Any:
    """
    Reduce the values in the dictionary from all processes so that all processes have the averaged results.

    :param input_dict: all the values will be reduced
    :param average: whether to do average or sum
    :return: a dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)  # type: ignore
        dist.all_reduce(values)  # type: ignore
        if average:
            values /= world_size  # type: ignore
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """Create logging metrics."""

    def __init__(self, delimiter: str = "\t") -> None:
        """Initialise the metric logger."""
        self.meters = defaultdict(SmoothedValue)  # type: ignore
        self.delimiter = delimiter

    def update(self, **kwargs: Any) -> None:
        """Update the metrics."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: Any) -> Any:
        """Get the requested stat attribute."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        """Metric logger representation."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        """Synchronise stats between the processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: Any, meter: Any) -> Any:
        """Add new meter to track."""
        self.meters[name] = meter

    def log_every(self, iterable: Any, print_freq: Any, header: Any = None) -> Any:
        """Log for every iterable."""
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,  # type: ignore
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Total time: ", total_time_str)
        #print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch: Any) -> Any:  # TODO: Remove!
    """Something."""
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer: Any, warmup_iters: Any, warmup_factor: Any) -> Any:
    """Warm up the learning rate scheduler."""

    def f(x: Any) -> Any:
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path: Any) -> Any:
    """Create directory."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master: Any) -> None:
    """Disable printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Check if available and successfully initialised."""
    if not dist.is_available():  # type: ignore
        return False
    if not dist.is_initialized():  # type: ignore
        return False
    return True


def get_world_size() -> int:
    """Get the number of processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()  # type: ignore


def get_rank() -> int:
    """Get the rank of the process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()  # type: ignore


def is_main_process() -> bool:
    """Check if the process is the main process."""
    return get_rank() == 0


def save_on_master(*args: Any, **kwargs: Any) -> None:
    """Save information on master process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args: Any) -> None:
    """Initialise regarding the distribution mode."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(  # type: ignore
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()  # type: ignore
    setup_for_distributed(args.rank == 0)
