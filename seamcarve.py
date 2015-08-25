import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter
import pyopencl as cl
import numpy as np
import time
import os


def min_argmin(l):
    """
        Returns the minimal element and its index.

    :param l: List
    :return: index, min value
    """
    ind, val = min(enumerate(l), key=itemgetter(1))
    return ind, val


def show_image(num, img, title, height, width, pth=None):
    """
        Plot the image.

    :param num: Image number
    :param img: Image
    :param title: Title
    :param height: height
    :param width: width
    :param pth: The seam path
    :return: None
    """
    if img is not None:
        plt.figure(num)
        plt.clf()
        plt.imshow(img)
        if pth is not None:
            plt.plot([i[1] for i in pth], [i[0] for i in pth], 'r-')
        plt.title(title)
        plt.xlim([0, width])
        plt.ylim([height, 0])


def get_energy_image(img):
    """
        Calculate the energy function as the discrete derivative of the image.

    :param img: Input image
    :return: The discrete derivatives for each pixel.
    """
    t = time.time()

    # select between PyOpenCL and single thread python implementation
    if USE_PYOPENCL:
        energy = opencl.get_energy(img)
    else:
        height, width = img.shape[:2]
        energy = np.zeros((height, width))

        for h in range(height):
            for w in range(width):
                w_l = max(0, w-1)
                w_r = min(w+1, width-1)
                h_u = max(0, h-1)
                h_d = min(h+1, height-1)
                energy[h, w] = sum(abs(img[h_u, w, :] - img[h_d, w, :])) + \
                               sum(abs(img[h, w_l, :] - img[h, w_r, :]))

    ENERGY_CALC_TIMES.append(time.time()-t)
    return energy


def find_seam_vertical(energy):
    """
        Back track ideology:
            -1   0    1
              \  |  /
                [ ]

    :param energy: Energy plot
    :return: List of seam coordinates (y, x) pairs.
    """
    t = time.time()

    height, width = energy.shape
    M = np.zeros(energy.shape, dtype=np.float32)
    M[0, :] = energy[0, :]
    backtrace = np.zeros(energy.shape, dtype=int)

    # select between PyOpenCL and single thread python implementation
    if USE_PYOPENCL:
        for h in range(1, height):
            M[h, :], backtrace[h, :] = opencl.find_seam(M[h-1, :], energy[h, :])
    else:
        backtrace[:, 0] = 1     # since first column does not have element to the left
        for h in range(1, height):
            for w in range(width):
                w_l = max(0, w-1)
                w_r = min(w+2, width)

                ind, val = min_argmin(M[h-1, w_l:w_r])
                M[h, w] = energy[h, w] + val
                backtrace[h, w] += ind - 1

    # backtrack to get the path
    ind, _ = min_argmin(M[-1, :])
    seam = [ind]
    for h in range(height-1, 0, -1):
        ind += backtrace[h, ind]
        seam.append(ind)
    seam.reverse()                  # order from top to bottom
    seam = list(enumerate(seam))    # transform to coordinates

    SEAM_SEARCH_TIMES.append(time.time()-t)
    return seam


def find_seam_horizontal(energy):
    """
        Back track ideology:
            -1
              \
            0 - []
              /
            1

    :param energy: Energy plot
    :return: List of seam coordinates (y, x) pairs.
    """
    t = time.time()

    height, width = energy.shape
    M = np.zeros(energy.shape)
    M[:, 0] = energy[:, 0]
    backtrace = np.zeros(energy.shape, dtype=int)

    # select between PyOpenCL and single thread python implementation
    if USE_PYOPENCL:
        for w in range(1, width):
            M[:, w], backtrace[:, w] = opencl.find_seam(M[:, w-1], energy[:, w])
    else:
        backtrace[0, :] = 1     # since first column does not have element up
        for w in range(1, width):
            for h in range(height):
                h_u = max(0, h-1)
                h_d = min(h+2, height)

                ind, val = min_argmin(M[h_u:h_d, w-1])
                M[h, w] = energy[h, w] + val
                backtrace[h, w] += ind - 1

    # backtrack to get the path
    ind, _ = min_argmin(M[:, -1])
    seam = [ind]
    for w in range(width-1, 0, -1):
        ind += backtrace[ind, w]
        seam.append(ind)
    seam.reverse()                                  # order from top to bottom
    seam = [(y, x) for x, y in enumerate(seam)]     # transform to coordinates

    SEAM_SEARCH_TIMES.append(time.time()-t)
    return seam


def remove_one_seam_from_image(img, trace, horizontal):
    """
        Remove the seam from the image.

    :param img: Image
    :param trace: Seam as a list of (y, x) coordinates.
    :param horizontal: Whether we remove horizontal or vertical seam.
    :return: Smaller image, the cost of reduction.
    """
    if horizontal:    # dirty hack, since reshaping does only work this way, otherwise the image is corrupted
        img = img.swapaxes(0, 1)
        trace = [(x, y) for y, x in trace]

    height, width, depth = img.shape
    mask = np.ones(img.shape, dtype=bool)
    cost = np.sum(img)

    # create the mask
    for h, w in trace:
        mask[h, w, :] = False

    # remove the pixels on the seam
    img = img[mask].reshape(height, width-1, depth)
    cost -= np.sum(img)

    # flip back for horizontal
    if horizontal:
        img = img.swapaxes(0, 1)

    return img, cost


def shrink_height(img):
    """
        Find the best horizontal seam to be removed from the image
        and return image without that seam.

    :param img: Input image
    :return: Image with one pixel less height.
    """
    eng = get_energy_image(img)
    path = find_seam_horizontal(eng)
    img, cost = remove_one_seam_from_image(img, path, horizontal=True)
    return img, eng, path, cost


def shrink_width(img):
    """
        Find the best vertical seam to be removed from the image
        and return image without that seam.

    :param img: Input image
    :return: Image with one pixel less width.
    """
    eng = get_energy_image(img)
    path = find_seam_vertical(eng)
    img, cost = remove_one_seam_from_image(img, path, horizontal=False)
    return img, eng, path, cost


def seam_carve(img, dw=0, dh=0):
    """
        Shrink image with seam-carving to desired size.

    :param img: Image
    :param dw: Shrink width for dw pixels
    :param dh: Shrink height for dh pixels
    :return: Smaller image
    """
    eng, path = None, None
    progress = 0
    final = 0

    def update_progress():
        nonlocal progress
        progress += 1
        print("\rProgress {:5.2f}%".format(100*progress/final), end='')
        if progress == final:
            print("\r", end='')

    if dh == 0 and dw == 0:
        return img, None, None

    elif dw == 0:   # remove just horizontal seams
        final = dh
        for i in range(dh):
            img, eng, path, _ = shrink_height(img)
            update_progress()
        return img, eng, path

    elif dh == 0:   # remove just vertical seams
        final = dw
        for i in range(dw):
            img, eng, path, _ = shrink_width(img)
            update_progress()
        return img, eng, path

    else:           # remove both
        final = 2*dw*dh + dh + dw
        REMOVE_HORIZONTAL = '^'
        REMOVE_VERTICAL = '<'

        # init
        img_map = np.zeros((dh+1, dw+1), dtype=object)      # used to store images
        backtrace = np.zeros((dh+1, dw+1), dtype=object)    # used to backtrack ream removal order
        T = np.zeros((dh+1, dw+1))                          # used to store the cost of optimal removal order

        # store the initial image
        img_map[0, 0] = img

        # fill horizontal border
        for i in range(1, dh+1):
            current_img = np.copy(img_map[i-1, 0])
            current_img, eng, path, cost = shrink_height(current_img)
            T[i, 0] = T[i-1, 0] + cost
            img_map[i, 0] = current_img
            backtrace[i, 0] = REMOVE_HORIZONTAL
            update_progress()

        # fill vertical border
        for i in range(1, dw+1):
            current_img = np.copy(img_map[0, i-1])
            current_img, eng, path, cost = shrink_width(current_img)
            T[0, i] = T[0, i-1] + cost
            img_map[0, i] = current_img
            backtrace[0, i] = REMOVE_VERTICAL
            update_progress()

        # use dynamic programing to find the best order
        for ih in range(1, dh+1):
            for iw in range(1, dw+1):
                # option 1: remove one horizontal seam from image above
                current_H = np.copy(img_map[ih-1, iw])
                current_H, eng_H, path_H, cost_H = shrink_height(current_H)
                cost_H += T[ih-1, iw]
                update_progress()

                # option 2: remove one vertical seam from image left
                current_V = np.copy(img_map[ih, iw-1])
                current_V, eng_V, path_V, cost_V = shrink_width(current_V)
                cost_V += T[ih, iw-1]
                update_progress()

                # select the optimal option
                if cost_H < cost_V:
                    T[ih, iw] = cost_H
                    img_map[ih, iw] = current_H
                    backtrace[ih, iw] = REMOVE_HORIZONTAL
                    eng, path = eng_H, path_H
                else:
                    T[ih, iw] = cost_V
                    img_map[ih, iw] = current_V
                    backtrace[ih, iw] = REMOVE_VERTICAL
                    eng, path = eng_V, path_V
        return img_map[-1, -1], eng, path


class PyOpenCLDriver:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = None

    def load_program(self, filename):
        f = open(filename, 'r')
        f_str = "".join(f.readlines())
        self.program = cl.Program(self.ctx, f_str).build()

    def get_energy(self, img):
        mf = cl.mem_flags

        H, W, D = map(np.int32, img.shape)
        img = img.astype(np.float32).reshape(-1)
        res = np.empty_like(img)

        img_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
        res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, res.nbytes)

        self.program.energy(self.queue, img.shape, None,
                            img_buf, res_buf, H, W, D)
        cl.enqueue_read_buffer(self.queue, res_buf, res).wait()

        res = res.reshape((H, W, D))
        return np.sum(res, axis=2)

    def find_seam(self, m, energy):
        mf = cl.mem_flags

        W = np.int32(m.shape[0])
        m = m.astype(np.float32)
        energy = energy.astype(np.float32)
        new_m = np.empty_like(m)
        back = np.empty_like(m)

        m_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
        energy_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=energy)
        new_m_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, new_m.nbytes)
        back_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, back.nbytes)

        self.program.find_seam(self.queue, m.shape, None,
                               m_buf, energy_buf, new_m_buf, back_buf, W)
        cl.enqueue_read_buffer(self.queue, new_m_buf, new_m).wait()
        cl.enqueue_read_buffer(self.queue, back_buf, back).wait()
        return new_m, back


#
#   Settings
#
USE_PYOPENCL = True     # If False, use pure python single-thread implementation, else use PyOpenCL
PLOT_RESULTS = True     # If True the results are also shown.
ENERGY_CALC_TIMES = []  # For time measurements
SEAM_SEARCH_TIMES = []  # For time measurements

opencl = PyOpenCLDriver()
opencl.load_program("get_energy.cl")
os.environ["PYOPENCL_CTX"] = "0:1"  # Select the device on which to run OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
print("Using: {}".format("PyOpenCL" if USE_PYOPENCL else "single CPU"))


if __name__ == "__main__":
    image = mpimg.imread(os.path.join('img', 'nature_1024.png'))
    original = np.copy(image)
    t0 = time.time()
    image, energy, path = seam_carve(image, dw=5, dh=5)
    t1 = time.time()

    print("Image shape:\n\t- original:\t\t{}\n\t- seam-carved:\t{}".format(
        original.shape, image.shape))
    print("Times:\n"
          "\t- one energy calc:\t{:.4f}s [avg of {}]\n"
          "\t- one seam search:\t{:.4f}s [avg of {}]\n"
          "\t- total:\t\t\t{:.4f}s".format(
        sum(ENERGY_CALC_TIMES)/len(ENERGY_CALC_TIMES),
        len(ENERGY_CALC_TIMES),
        sum(SEAM_SEARCH_TIMES)/len(SEAM_SEARCH_TIMES),
        len(SEAM_SEARCH_TIMES),
        t1-t0))

    # plot
    if PLOT_RESULTS:
        H, W = original.shape[:2]
        show_image(1, original, "Original", H, W)
        show_image(2, energy, "Last seam on energy plot", H, W, path)
        show_image(3, image, "Seam carving", H, W)
        plt.show()