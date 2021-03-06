import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

D = 30
d = 29
N = 3000
r = 0.1    # change r = 0.1 to plot a; r = 0.2 to plot b
M = int(r * N / (1 - r))
num_seg = 100
sigma_limit = 0.1

if not os.path.isdir("./files"):
    os.system('mkdir files')
# subprocess.call(['/bin/bash', '-i', '-c', 'g++ -std=c++11 -I /home/tanglw/Downloads/armadillo-10.2.1/include/ -DARMA_DONT_USE_WRAPPER  generate_fig_8.cpp -O2 -o generate_fig_8 -lopenblas '])
# os.system('./generate_fig_8 -D %d -d %d -N %d -r %f -num %d -sig %f' % (D, d, N, r, num_seg, sigma_limit))

filename = ['theta2.ty', 'hat_theta2.ty']

data = dict()
for file in filename:
    with open('./files/' + file, 'r') as f:
        vals = f.readlines()
    data[file] = []
    for each in vals:
        data[file].append(float(each.strip()))
    data[file] = np.array(data[file])


sigma = np.linspace(0, sigma_limit, num_seg)

ff = plt.figure(figsize=(10, 8))
plt.plot(sigma, data['theta2.ty'], 'r--', sigma, data['hat_theta2.ty'], 'g', linewidth=4)
plt.xlabel(r'$\sigma$', fontsize=20)
plt.ylabel('angle', fontsize=20)
plt.legend([r'$sin^{-1}(t_2)$', r'$\gamma$'], fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks([0, 0.05, 0.1])

plt.show()
ff.savefig("foo.pdf", bbox_inches='tight')
