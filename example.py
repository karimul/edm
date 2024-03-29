# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib

#----------------------------------------------------------------------------
def adjust_learning_rate(iter:int, total_epoch:int=100, M:int=3, lr0:float=0.007):
    cos_inner = np.pi * (iter % (total_epoch // M))
    cos_inner /= total_epoch // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr0
    return lr

def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, cyclical=False,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    total_epochs = len(list(enumerate(zip(t_steps[:-1], t_steps[1:]))))**2
    print("total epoch:", total_epochs)
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        if cyclical is True:
            # Adaptive step size of noise with cyclical
            S_noise_new = adjust_learning_rate(i, total_epoch=total_epochs, lr0=S_noise)
            x_hat = x_cur + S_noise_new * torch.randn_like(x_cur) 
            # x_hat = x_cur + S_noise_new * torch.randn_like(x_cur) * (t_hat ** 2 - t_cur ** 2).sqrt()
            # print(f"{(t_hat ** 2 - t_cur ** 2).sqrt()} {S_noise_new} {t_hat} {t_cur}")
        else:
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) 
        

    # Save image grid.
    dest_path = f"{num_steps}_{cyclical}_{dest_path}"
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')

#----------------------------------------------------------------------------

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    seeds=50000-99999
    generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-ve.pkl',   '5_cifar10-32x32.png',  num_steps=3, S_noise=0.0001, cyclical=True, seed=seeds) # FID = 1.79, NFE = 35
    generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-ve.pkl',   '15_cifar10-32x32.png',  num_steps=8, S_noise=0.0001, cyclical=True, seed=seeds) # FID = 1.79, NFE = 35
    generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-ve.pkl',   '25_cifar10-32x32.png',  num_steps=13, S_noise=0.0001, cyclical=True, seed=seeds) # FID = 1.79, NFE = 35
    generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-ve.pkl',   '35_cifar10-32x32.png',  num_steps=18, S_noise=0.0001, cyclical=True, seed=seeds) # FID = 1.79, NFE = 35
    # generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=35, S_churn=0.001, S_noise=0.01, cyclical=True, seed=seeds) # FID = 2.39, NFE = 79
    # generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=35, seed=seeds) # FID = 2.39, NFE = 79
    # generate_image_grid(f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl',  'afhqv2-64x64.png',   num_steps=35, S_churn=0.001, S_noise=0.01, cyclical=True, seed=seeds) # FID = 1.96, NFE = 79
    # generate_image_grid(f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl',  'afhqv2-64x64.png',   num_steps=35, seed=seeds) # FID = 1.96, NFE = 79
    # generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=10, S_noise=0.001, cyclical=True) # FID = 1.36, NFE = 511
    # generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=256) # FID = 1.36, NFE = 511

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
