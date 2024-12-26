import os
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.distributed as dist
from diffusion import create_gaussian_diffusion
from torch import nn
from tqdm import tqdm
import numpy as np
from data.keypoint import KeyDataset
from unet import UNetModel
from ddim import DDIMSampler



def init_distributed():
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_main_process():
    try:
        if dist.get_rank() == 0:
            return True
        else:
            return False
    except:
        return True


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def test(dataloader, model, diffusion,path='/zy/image'):
    i = 0
    ddim = DDIMSampler(model=diffusion)
    model.eval()
    for data in tqdm(dataloader):
        with torch.no_grad():
            i = i + 1
            diffusion.set_input(data, batch_size=1)
            #diffusion.ddim_Sample(ddim,model,i,path)
            diffusion.Sample(model,i,path)







def main():
    load_model = True
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.manual_seed(20)
    test_dataset = KeyDataset()
    test_dataset.initialize(phase='test')

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=16,
        sampler=test_sampler)

    model = UNetModel()
    model = model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    betas = (0.0001, 0.02)
    n_T = 1000
    beta_schedule = generate_linear_schedule(n_T, betas[0] * 1000 / n_T, betas[1] * 1000 / n_T, )
    diffusion = create_gaussian_diffusion(beta_schedule, predict_xstart=False)

    if load_model:
        checkpoint = torch.load("/zy/diffusion1/model_576_86.28188375785248_6.93599665327929.pth", map_location='cpu')
        model.module.load_state_dict(checkpoint['net'])
    test(test_loader, model, diffusion)



if __name__ == "__main__":
    init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main()