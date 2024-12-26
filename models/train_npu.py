import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch_npu
import torch.distributed as dist
from diffusion import create_gaussian_diffusion
from torch import nn
from tqdm import tqdm
import numpy as np
from data.keypoint import KeyDataset
from unet import UNetModel
from tensorfn.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP



def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map



def main_worker(dev, npus_per_node, args):
    device_id = args.process_device_map[dev]
    loc = 'npu:{}'.format(device_id)
    if args.device == 'npu':
        torch.npu.set_device(loc)
        print("Use NPU: {} for training".format(device_id))
    device = torch.device(loc) if args.device == 'npu' else torch.device('cpu')

    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.global_rank == -1:
            args.global_rank = int(os.environ["RANK"])
        args.global_rank = args.global_rank * npus_per_node + dev

        dist.init_process_group(backend=args.dist_backend,  # init_method=cfg.dist_url,
                                world_size=args.world_size, rank=args.global_rank)

    load_model = False
    torch.manual_seed(20)
    train_dataset = KeyDataset()
    train_dataset.initialize(phase='train')
    test_dataset = KeyDataset()
    test_dataset.initialize(phase='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=False, drop_last=True, pin_memory=False, num_workers=16,
        sampler=train_sampler)
    SUM = int(((len(train_dataset) / 16)) * 0.3)

    model = UNetModel()
    model = model.to(args.device)
    model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)

    lrate = 2e-5
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    # scheduler = lr_scheduler.cycle_scheduler(optimizer=optim,lr=2e-5,n_iter=2400000,warmup=3814*3,decay=("linear","flat"))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.96)
    betas = (0.0001, 0.02)
    n_T = 1000
    beta_schedule = generate_linear_schedule(n_T, betas[0] * 1000 / n_T, betas[1] * 1000 / n_T, )
    diffusion = create_gaussian_diffusion(beta_schedule, predict_xstart=False)

    if load_model:
        checkpoint = torch.load("/zy/diffusion1/model_11_34.25419665573281.pth", map_location='cpu')
        model.module.load_state_dict(checkpoint['net'])
        optim.load_state_dict(checkpoint['optimizer'])
        print(checkpoint['lr'])


    train(train_loader, model, diffusion, optim, scheduler, SUM,args, device)


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



def train(dataloader, model, diffusion, optimizer, scheduler, SUM,args,device):
    import time

    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
    scheduler.step()
    for epoch in range(12, 1000):
        dataloader.sampler.set_epoch(epoch)
        if is_main_process:
            print('#Epoch - ' + str(epoch))
            print('SUM', SUM)
        start_time = time.time()
        sum_loss = 0
        i = 0
        model.train()
        for data in tqdm(dataloader):
            i = i + 1
            diffusion.set_input(data, batch_size=8)
            loss_dict = diffusion.training_losses(model, i, SUM)

            loss = loss_dict['loss'].mean()
            # loss_mse = loss_dict['mse'].mean()
            # loss_vb = loss_dict['vb'].mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss = loss_dict['loss'].mean()
            loss_list.append(loss.detach().item())
            # loss_mean_list.append(loss_mse.detach().item())
            # loss_vb_list.append(loss_vb.detach().item())
            sum_loss += loss.detach().item()

            # if is_main_process() and i %2000==0:
            #     print('Epoch Time ' + str(int(time.time() - start_time)) + ' secs')
            #     print('Model Saved Successfully for #epoch ' + str(epoch) + ' #steps ' + str(i))

            # state_dict = {"net": (model.module).state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            #               "lr": optimizer.param_groups[0]['lr']}
            #
            # torch.save(state_dict, '/zy/diffusion1/' + f"model_{epoch}_{sum_loss}.pth")

            print('loss:', loss.detach().item(), 'sum_loss', sum_loss, 'aver_loss', sum_loss / (i))
            print('lr', optimizer.param_groups[0]['lr'])
        scheduler.step()

        if is_main_process():
            print('Epoch Time ' + str(int(time.time() - start_time)) + ' secs')
            print('Model Saved Successfully for #epoch ' + str(epoch) + ' #steps ' + str(i))

            state_dict = {"net": model.module.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                          "lr": optimizer.param_groups[0]['lr']}

            torch.save(state_dict, '/zy/diffusion1/' + f"model_{epoch}_{sum_loss}.pth")


def main(args):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29501'
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    if args.device == 'npu':
        npus_per_node = len(args.process_device_map)
    else:
        npus_per_node = torch.npu.device_count()
    print('{} node found.'.format(npus_per_node))
    if args.multiprocessing_distributed:
        args.world_size = npus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, args))
    else:
        main_worker(0, npus_per_node, args)





if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)


    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
    parser.add_argument('--device_list', default='0,1', type=str, help='device id list')
    parser.add_argument('--dist_backend', default='hccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()

    main(args)