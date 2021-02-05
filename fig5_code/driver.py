import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import time

D = 5
d = 3
N = 500
r = 0.6
M = int(r * N / (1 - r))
num_seg = 1
sigma = 1e-3    # change sigma = 0 to plot a and sigma = 1e-6 to plot b

if not os.path.isdir("./files"):
    os.system('mkdir files')
# subprocess.call(['/bin/bash', '-i', '-c', 'g++ -std=c++11 -larmadillo generate_fig5.cpp -O2 -o generate_fig5 -DNDEBUG -framework Accelerate'])
# subprocess.call(['/bin/bash', '-i', '-c', 'g++ -std=c++11 -I /home/tanglw/Downloads/armadillo-10.2.1/include/ -DARMA_DONT_USE_WRAPPER  generate_fig5.cpp -O2 -o generate_fig5 -lopenblas '])

t1 = time.time()
os.system('./generate_fig5 -D %d -d %d -N %d -r %f -num %d -sig %f' % (D, d, N, r, num_seg, sigma))
print(time.time()-t1)
fileOpt1 = ['tan_theta.ty', 'stepsize.ty', 'bound.ty']

filename = fileOpt1

data = dict()
for file in filename:
    with open("./files/" + file, 'r') as f:
        vals = f.readlines()
    data[file] = []
    for each in vals:
        data[file].append(float(each.strip()))
    data[file] = np.array(data[file])

iter = np.linspace(1, 5000, 5000)
if filename == fileOpt1:
    fig = plt.figure(figsize=(10, 8))

    plt.plot(iter, data['bound.ty'], 'r:', linewidth=3)
    plt.plot(iter, data['tan_theta.ty'], 'b', linewidth=3)
    plt.plot(iter, data['stepsize.ty'], 'g-.', linewidth=3)

    plt.legend(['upper bound', r'$\tan(\theta_k)$', r'$\mu_k$'], fontsize=20, loc='best')

    plt.xlabel('iteration', fontsize=20)
    plt.yscale('log')
    plt.xticks([0, 2500, 5000], fontsize=20)
    plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9], fontsize=20)
    plt.axis([0, 5000, 1e-10, 10**(-0.5)])

    fig.savefig("foo.pdf", bbox_inches='tight')


plt.show()
