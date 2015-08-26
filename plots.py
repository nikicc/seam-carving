import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seamcarve
import time
import os

FOLDER = os.path.join('results')
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

NAMES = [
    '32x20',
    '64x40',
    '128x80',
    '256x160',
    '512x320',
    '1024x640'
]

CPU_TOTAL = []
CPU_ENERGY = []
CPU_SEAM = []
CL_CPU_TOTAL = []
CL_CPU_ENERGY = []
CL_CPU_SEAM = []
CL_GPU_TOTAL = []
CL_GPU_ENERGY = []
CL_GPU_SEAM = []
NUM = 25

for flag, device, TOTAL, ENERGY, SEAM in (
        (False, "", CPU_TOTAL, CPU_ENERGY, CPU_SEAM),
        (True, "0:0", CL_CPU_TOTAL, CL_CPU_ENERGY, CL_CPU_SEAM),
        (True, "0:1", CL_GPU_TOTAL, CL_GPU_ENERGY, CL_GPU_SEAM),
):
    for IMG in [
        'nature_32.png',
        'nature_64.png',
        'nature_128.png',
        'nature_256.png',
        'nature_512.png',
        'nature_1024.png'
    ]:
        seamcarve.USE_PYOPENCL = flag
        os.environ["PYOPENCL_CTX"] = device
        print('Image:{}\tDevice:{}'.format(IMG, device))

        seamcarve.ENERGY_CALC_TIMES = []  # For time measurements
        seamcarve.SEAM_SEARCH_TIMES = []  # For time measurements

        image = mpimg.imread(os.path.join('img', IMG))
        t0 = time.time()
        image, energy, path = seamcarve.seam_carve(image, dw=NUM, dh=0)
        t1 = time.time()

        TOTAL.append((t1-t0)/NUM)
        ENERGY.append(sum(seamcarve.ENERGY_CALC_TIMES)/len(seamcarve.ENERGY_CALC_TIMES))
        SEAM.append(sum(seamcarve.SEAM_SEARCH_TIMES)/len(seamcarve.SEAM_SEARCH_TIMES))

plt.figure(1, figsize=(6, 4.5))
plt.clf()
plt.plot(range(len(NAMES)), CPU_TOTAL, 'r', label='CPU Python')
plt.plot(range(len(NAMES)), CL_GPU_TOTAL, 'b', label='GPU Iris Pro')
plt.plot(range(len(NAMES)), CL_CPU_TOTAL, 'g', label='CPU PyOpenCL')
plt.xticks(range(len(NAMES)), NAMES, size='small')
plt.title("Total time for removing one vertical seam")
plt.ylabel('Total time [s]')
plt.xlabel('Image size [pixels]')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig(os.path.join(FOLDER, 'total-time.pdf'))

plt.figure(2, figsize=(6, 4.5))
plt.clf()
plt.plot(range(len(NAMES)), CPU_ENERGY, 'r', label='CPU Python')
plt.plot(range(len(NAMES)), CL_GPU_ENERGY, 'b', label='GPU Iris Pro')
plt.plot(range(len(NAMES)), CL_CPU_ENERGY, 'g', label='CPU PyOpenCL')
plt.xticks(range(len(NAMES)), NAMES, size='small')
plt.title("Time for energy calculation")
plt.ylabel('Total time [s]')
plt.xlabel('Image size [pixels]')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig(os.path.join(FOLDER, 'energy-time.pdf'))

plt.figure(3, figsize=(6, 4.5))
plt.clf()
plt.plot(range(len(NAMES)), CPU_SEAM, 'r', label='CPU Python')
plt.plot(range(len(NAMES)), CL_GPU_SEAM, 'b', label='GPU Iris Pro')
plt.plot(range(len(NAMES)), CL_CPU_SEAM, 'g', label='CPU PyOpenCL')
plt.xticks(range(len(NAMES)), NAMES, size='small')
plt.title("Time for finding a seam")
plt.ylabel('Total time [s]')
plt.xlabel('Image size [pixels]')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig(os.path.join(FOLDER, 'seam-time.pdf'))
#plt.show()


with open(os.path.join(FOLDER, 'measurements.txt'), 'w') as f:
    for L in ['CPU_TOTAL', 'CPU_ENERGY', 'CPU_SEAM',
              'CL_CPU_TOTAL', 'CL_CPU_ENERGY', 'CL_CPU_SEAM',
              'CL_GPU_TOTAL', 'CL_GPU_ENERGY', 'CL_GPU_SEAM']:
        f.write('{} = {}\n'.format(
            L,
            [float("{:.5f}".format(i)) for i in globals()[L]]
        ))