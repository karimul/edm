import subprocess

f = open("output.txt", "w")

# CIFAR10
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


# # AFHQ V2
# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-4999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=5000-9999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=10000-14999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz", shell=True, check=True, stdout=f)

# # FFHQ
# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-4999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=5000-9999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=10000-14999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=40 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz", shell=True, check=True, stdout=f)

# # ImageNet
# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-4999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=5000-9999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)

# subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
# subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=10000-14999 --subdirs \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
#     --cyclical --S_churn=0.0001 --S_noise=0.0001 --steps=256 --sigma_max=50", shell=True, check=True, stdout=f)
# subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
#     --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz", shell=True, check=True, stdout=f)