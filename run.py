import subprocess

f = open("output.txt", "w")
subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_noise=10 --steps=18 --sigma_max=60", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=50000-99999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_noise=10 --steps=18 --sigma_max=60", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)

subprocess.run("rm -rf fid-tmp/", shell=True, check=True)
subprocess.run("torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=100000-149999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl \
    --cyclical --S_noise=10 --steps=18 --sigma_max=60", shell=True, check=True, stdout=f)
subprocess.run("torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz", shell=True, check=True, stdout=f)