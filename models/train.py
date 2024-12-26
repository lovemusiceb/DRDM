import os
import warnings

from contextlib import nullcontext
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from diffusion import create_gaussian_diffusion
from torch import nn
from tqdm import tqdm
import numpy as np
from data.keypoint import KeyDataset
from unet import UNetModel
from tensorfn.optim import lr_scheduler
from torch.backends import cudnn


class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model.module if type(model) is torch.nn.DataParallel else model)
        self.model.eval()

    def update_fn(self, model, fn):
        with torch.no_grad():
            model = model.module if type(model) is torch.nn.DataParallel else model
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(fn(ema_v, model_v))

    def update(self, model):
        self.update_fn(model, fn=lambda e, m: self.decay * e + (1. - self.decay) * m)





def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

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
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True





def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)



def train(dataloader, model, diffusion, optimizer, scheduler,SUM,testloader):

    import time
    scheduler.step()
    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
    for epoch in range(574,1000):

        dataloader.sampler.set_epoch(epoch)
        if is_main_process:
            print ('#Epoch - '+str(epoch))
            print('SUM', SUM)
        start_time = time.time()
        sum_loss=0
        sum_loss_1=0
        i=0
        model.train()
        optimizer.zero_grad()
        for data in tqdm(dataloader):
            i = i + 1
            my_context = model.no_sync if  i % 3 != 0 else nullcontext
            with my_context():
                diffusion.set_input(data, batch_size=7)
                loss_dict = diffusion.training_losses(model, i, SUM)
                loss = loss_dict['loss'].mean()
                loss = loss /3
                loss.backward()
            if i % 3 == 0:
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            k=loss.detach().item()*3
            if k<=0.01:
                sum_loss_1 += k
                sum_loss+=k
            else:
                sum_loss += k
                sum_loss_1+=0.01
            # if is_main_process() and i %2000==0:
            #     print('Epoch Time ' + str(int(time.time() - start_time)) + ' secs')
            #     print('Model Saved Successfully for #epoch ' + str(epoch) + ' #steps ' + str(i))

                # state_dict = {"net": (model.module).state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                #               "lr": optimizer.param_groups[0]['lr']}
                #
                # torch.save(state_dict, '/zy/diffusion1/' + f"model_{epoch}_{sum_loss}.pth")

            print('loss:', loss.detach().item()*3, 'sum_loss',sum_loss, 'aver_loss', sum_loss / (i))
            print('lr',optimizer.param_groups[0]['lr'])
        #scheduler.step()



        sun_test=0
        ema_model = model.eval()
        with torch.no_grad():
            testloader.sampler.set_epoch(epoch)
            for data in tqdm(testloader):
                diffusion.set_input(data, batch_size=7)
                loss_dict = diffusion.training_losses(ema_model, i, SUM)
                test_loss = loss_dict['loss'].mean()
                sun_test+=test_loss.detach().item()




        if is_main_process():

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')
            print ('Model Saved Successfully for #epoch '+str(epoch)+' #steps '+str(i))

            state_dict = {"net": ema_model.module.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                          "lr": optimizer.param_groups[0]['lr']}

            torch.save(state_dict, '/zy/diffusion1/' + f"model_{epoch}_{sum_loss}_{sun_test}.pth")





def main():



    load_model=True
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.manual_seed(20)
    train_dataset = KeyDataset()
    train_dataset.initialize(phase='train')
    test_dataset = KeyDataset()
    test_dataset.initialize(phase='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=7, shuffle=False, drop_last=True, pin_memory=True, num_workers=128,
        sampler=train_sampler)
    SUM = int(((len(train_dataset)/16))*0.3)



    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    testloader=torch.utils.data.DataLoader(
        train_dataset, batch_size=7, shuffle=False, drop_last=False, pin_memory=True, num_workers=128,
        sampler=test_sampler)



    model =UNetModel()
    model = model.to(args.device)
    #
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total / 1e6))

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )


    lrate=2e-5
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    # scheduler  = lr_scheduler.cycle_scheduler(optimizer=optim,lr=2e-5,n_iter=10000000,warmup=4359,decay=("linear","flat"))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.0001)
    betas = (0.0001, 0.02)
    n_T = 1000
    beta_schedule = generate_linear_schedule(n_T, betas[0] * 1000 / n_T, betas[1] * 1000 / n_T, )
    diffusion = create_gaussian_diffusion(beta_schedule, predict_xstart = False)

    if load_model:
        checkpoint = torch.load("/zy/diffusion1/model_573_76.1264423224493_34.99898810842912.pth", map_location='cpu')
        model.module.load_state_dict(checkpoint['net'])
        optim.load_state_dict(checkpoint['optimizer'])
        print(checkpoint['lr'])



    train(train_loader,model,diffusion,optim,scheduler,SUM,testloader)



if __name__ == "__main__":

    init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)


    args = parser.parse_args()


    main()
