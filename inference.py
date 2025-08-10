import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from thop import profile
from thop import clever_format
import basicsr.archs.dmnet_arch as network


if __name__ == '__main__':
    upscale = 2
    height = (1280 // upscale)
    width = (720 // upscale)
    model = network.DMNet(upscale=upscale)

    input = torch.randn(1, 3, height, width)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    macs, params = profile(model.to(device), inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs:", macs)
    print("params:", params)



    # cudnn.benchmark = True

    # device = 'cpu:0' #'cuda:0'

    # repetitions = 100

    # model = model.to(device)
    # dummy_input = torch.rand(1, 3, height, width).to(device)

    # # warm up
    # # print('warm up ...\n')
    # # with torch.no_grad():
    # #     for _ in range(100):
    # #         _ = model(dummy_input)

    # # synchronize / wait for all the GPU process then back to cpu
    # torch.cuda.synchronize()

    # # testing CUDA Event
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # initialize
    # timings = np.zeros((repetitions, 1))

    # print('testing ...\n')
    # with torch.no_grad():
    #     for rep in tqdm.tqdm(range(repetitions)):
    #         starter.record()
    #         _ = model(dummy_input)
    #         ender.record()
    #         torch.cuda.synchronize()  # wait for ending
    #         curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
    #         timings[rep] = curr_time

    # avg = timings.sum() / repetitions
    # print('\navg={}\n'.format(avg))