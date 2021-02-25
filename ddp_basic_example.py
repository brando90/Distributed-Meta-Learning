"""
Based on: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Correctness of code: https://stackoverflow.com/questions/66226135/how-to-parallelize-a-training-loop-ever-samples-of-a-batch-when-cpu-is-only-avai

Note: as opposed to the multiprocessing (torch.multiprocessing) package, processes can use
different communication backends and are not restricted to being executed on the same machine.
"""
import time

from typing import Tuple

import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os

num_epochs = 5
batch_size = 8
Din, Dout = 10, 5
data_x = torch.randn(batch_size, Din)
data_y = torch.randn(batch_size, Dout)
data = [(i*data_x, i*data_y) for i in range(num_epochs)]

class PerDeviceModel(nn.Module):
    """
    Toy example for a model ran in parallel but not distributed accross gpus
    (only processes with their own gpu or hardware)
    """
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(Din, Din)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(Din, Dout)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup_process(rank, world_size, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.
    """
    # set up the master's ip address so this child process can coordinate
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
    if torch.cuda.is_available():
        backend = 'nccl'
    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """ Destroy a given process group, and deinitialize the distributed package """
    dist.destroy_process_group()

def get_batch(batch: Tuple[torch.Tensor, torch.Tensor], rank):
    x, y = batch
    if torch.cuda.is_available():
        x, y = x.to(rank), y.to(rank)
    else:
        x, y = x.share_memory_(), y.share_memory_()
    return x, y

def get_batch_serial(batch: Tuple[torch.Tensor, torch.Tensor], rank):
    x, y = batch
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, y = x.to(device), y.to(device)
        # x, y = x.cuda(), y.cuda()
        # mdl.cuda() or mdl.to(device)
    else:
        x, y = x, y
    return x, y

def get_ddp_model(model: nn.Module, rank):
    """
    Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.

    :return:

    TODO: does this have to be done outside or inside the process? my guess is that it doesn't matter because
    1) if its on gpu once it's on the right proc it moves it to cpu with id rank via mdl.to(rank)
    2) if it's on cpu then mdl.share_memory() or data.share_memory() is a no op if it's already in shared memory o.w.
    """
    # if gpu avail do the standard of creating a model and moving the model to the GPU with id rank
    if torch.cuda.is_available():
    # create model and move it to GPU with id rank
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    else:
    # if we want multiple cpu just make sure the model is shared properly accross the cpus with shared_memory()
    # note that op is a no op if it's already in shared_memory
        model = model.share_memory()
        ddp_model = DDP(model)  # I think removing the devices ids should be fine...?
    return ddp_model
    # return OneDeviceModel().to(rank) if torch.cuda.is_available() else OneDeviceModel().share_memory()

def run_parallel_training_loop(rank, world_size):
    """
    Distributed function to be implemented later.

    This is the function that is actually ran in each distributed process.

    Note: as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
    you don’t need to worry about different DDP processes start from different model parameter initial values.
    """
    setup_process(rank, world_size)
    print()
    print(f"Start running DDP with model parallel example on rank: {rank}.")
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')

    # get ddp model
    model = PerDeviceModel()
    ddp_model = get_ddp_model(model, rank)

    # do training
    for batch_idx, batch in enumerate(data):
        x, y = get_batch(batch, rank)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(x)
        # Gradient synchronization communications take place during the backward pass and overlap with the backward computation.
        loss_fn(outputs, y).backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
        optimizer.step()  # TODO how does the optimizer know to do the gradient step only once?

    print()
    print(f"End running DDP with model parallel example on rank: {rank}.")
    print(f'End current process: {mp.current_process()}')
    print(f'End pid: {os.getpid()}')
    # Destroy a given process group, and deinitialize the distributed package
    cleanup()

def main():
    print()
    print('running main()')
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')
    # args
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = mp.cpu_count()
    world_size = 1
    print(f'world_size={world_size}')
    mp.spawn(run_parallel_training_loop, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    print('starting __main__')
    start = time.time()
    main()
    print(f'execution length = {time.time() - start}')
    print('Done!\a\n')
