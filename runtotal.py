import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle

project = sys.argv[1]
#seed = 0
seed = int(sys.argv[2])
#lr = 1e-2
lr = float(sys.argv[3])
#batch_size = 60
batch_size = int(sys.argv[4])
#NNName="GGAT"
NNName = sys.argv[5]
eps = int(sys.argv[6])
layer_size = int(sys.argv[7])
card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {"Lang":1,"Chart":1,"Math":1,"Cli":1,"JxPath":1,"Time":1}
K_size = {"Lang":1,"Chart":1,"Math":1,"Cli":1,"JxPath":1,"Time":1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum

K=len(lst)/K_size[project]
print(len(lst),K)

for i in tqdm(range(int(K/totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        k = i * totalnum + j
        if k>=K:
            continue
        cardn =int(j / singlenum)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python run.py %d %s %f %d %d %d %d %d %d %s"%(k, project, lr, seed, batch_size,K_size[project],len(lst),eps,layer_size,NNName), shell=True)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python sum.py %s %d %f %d %s %d"%(project, seed, lr, batch_size, str(layer_size)+'-'+NNName,eps), shell=True,)
p.wait()

## subprocess.Popen("python watch.py %s %d %f %d"%(project, seed, lr, batch_size,K_size[project]),shell=True)