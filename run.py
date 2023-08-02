import subprocess

# CIFAR10 Conditional VE
f = open("output_cifar10-32x32-cond-ve.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)
f.close()


# CIFAR10 Conditional VP
f = open("output_cifar10-32x32-cond-vp.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)
f.close()

# CIFAR10 UnConditional VE
f = open("output_cifar10-32x32-uncond-ve.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)
f.close()


# CIFAR10 UnConditional VP
f = open("output_cifar10-32x32-uncond-vp.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=18 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)
f.close()

# ImageNet
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz", shell=True, check=True, stdout=f)
f.close()

# AFHQ V2 VE
f = open("output_afhqv2-64x64-uncond-ve.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)
f.close()

# AFHQ V2 VP
f = open("output_afhqv2-64x64-uncond-ve.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)
f.close()

# FFHQ VP
f = open("output_ffhq-64x64-uncond-vp.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)
f.close()

# FFHQ VE
f = open("output_ffhq-64x64-uncond-vp.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl \
    --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)
f.close()

