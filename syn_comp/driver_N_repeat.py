import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import math


D = 4
d = 3
N = 140
r = 0.65
M = int(r * N / (1 - r))
num_seg = 40
sigma_limit = 0.04
n_step = 2


if not os.path.isdir("./files"):
    os.system('mkdir files')
# subprocess.call(['/bin/bash', '-i', '-c', 'g++ -std=c++11 -I /home/tanglw/Downloads/armadillo-10.2.1/include/ -DARMA_DONT_USE_WRAPPER  compare_PSGD_REAPER_GGD_N_M.cpp -O2 -o comp_N_M -lopenblas '])
os.system('./comp_N_M -D %d -d %d -N %d -r %f -num %d -sig %f' % (D, d, N, r, num_seg, sigma_limit))

# fileOpt1 = ['cos_phi_PSGD.ty', 'cos_phi_REAPER.ty', 'cos_phi_GGD.ty', 
        # 'cos_phi_PSGD_IRLS.ty', 'cos_phi_PSGD_IRLS_c1.ty']
fileOpt1 = ['cos_phi_PSGD.ty', 'cos_phi_REAPER.ty', 'cos_phi_GGD.ty', 
        'cos_phi_PSGD_IRLS.ty', 'cos_phi_PSGD_IRLS_c1.ty']

filename = fileOpt1

data = dict()
for file in filename:
    with open('./files/' + file, 'r') as f:
        vals = f.readlines()
    data[file] = []
    for each in vals:
        data[file].append(float(each.strip()))
    data[file] = np.array(data[file])


sigma = np.linspace(0, sigma_limit, num_seg)
ratios = np.linspace(0, r, num_seg)
Ns = [N+i*n_step for i in range(num_seg)]

if filename == fileOpt1:
    fig = plt.figure(figsize=(18, 8))
    plt.plot(Ns, np.log(90 - np.arccos(data['cos_phi_PSGD.ty'])*180/math.pi), 'r', linewidth=3)
    plt.plot(Ns, np.log(90 -np.arccos(data['cos_phi_REAPER.ty'])*180/math.pi), 'g', linewidth=3)
    plt.plot(Ns, np.log(90 -np.arccos(data['cos_phi_GGD.ty'])*180/math.pi), 'k', linewidth=3)
    plt.plot(Ns, np.log(90 -np.arccos(data['cos_phi_PSGD_IRLS.ty'])*180/math.pi), 'y', linewidth=3)

    # plt.plot(ratios, np.arccos(data['cos_phi_PSGD_IRLS_c1.ty'])*180/math.pi, 'c', linewidth=3)
    plt.xlabel(r'$ratios$', fontsize=15)
    plt.ylabel(r'$log(\phi_*)$', fontsize=15)
    # plt.legend(['PSGD', 'REAPER', 'GGD','PSGD_IRLS', 'PSGD_IRLS_c1'], fontsize=15)
    plt.legend(['PSGD', 'FMS', 'GGD','PSGD_IRLS'], fontsize=15)

    # plt.xticks(ratios, fontsize=15)
    # plt.yticks([85, 86, 87, 88,89, 90], fontsize=15)

    fig.savefig("foo.pdf", bbox_inches='tight')

plt.show()
