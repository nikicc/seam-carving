import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter
import pyopencl as cl
import numpy as np
import os


def min_argmin(l):
    """
    Returns the minimal element and its index.

    :param l: List
    :return: index, min value
    """
    ind, val = min(enumerate(l), key=itemgetter(1))
    return ind, val


def show_image(num, img, title, height, width, path=None):
    if img is not None:
        plt.figure(num)
        plt.clf()
        plt.imshow(img)
        if path is not None:
            plt.plot([i[1] for i in path], [i[0] for i in path], 'r-')
        plt.title(title)
        plt.xlim([0, width])
        plt.ylim([height, 0])


def get_energy_image(img):
    """
    Calculate the energy function as the discrete derivative of the image.

    :param img: Input image
    :return: The discrete derivatives for each pixel.
    """
    """
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
    """
    r = example.opencl_energy(img)
    #assert np.allclose(energy, r)
    return r


def find_seam_vertical(energy):
    """
    Back track ideology:
        -1   0    1
          \  |  /
            [ ]

    :param energy: Energy plot
    :return: List of seam coordinates (y, x) pairs.
    """

    height, width = energy.shape
    M = np.zeros(energy.shape)
    M[0, :] = energy[0, :]
    backtrace = np.zeros(energy.shape, dtype=int)
    backtrace[:, 0] = 1     # since first column does not have element to the left

    # find minimal energy seam
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
    seam.reverse()  # order from top to bottom
    return list(enumerate(seam))


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

    height, width = energy.shape
    M = np.zeros(energy.shape)
    M[:, 0] = energy[:, 0]
    backtrace = np.zeros(energy.shape, dtype=int)
    backtrace[0, :] = 1     # since first column does not have element up

    # find minimal energy seam
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
    seam.reverse()  # order from top to bottom
    return [(y, x) for x, y in enumerate(seam)]


def remove_one_seam_from_image(img, trace, horizontal):
    """
    Remove the seam from the image.

    :param img: Image
    :param path: Path as a list of (y, x) coordinates.
    :param vertical: Whether we remove vertical or horizontal seam.
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
        img_map = np.zeros((dh+1, dw+1), dtype=object)  # used to store images
        backtrace = np.zeros((dh+1, dw+1), dtype=object)   # used to backtrack ream removal order
        T = np.zeros((dh+1, dw+1))                      # used to store the cost of optimal removal order

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
                else:
                    T[ih, iw] = cost_V
                    img_map[ih, iw] = current_V
                    backtrace[ih, iw] = REMOVE_VERTICAL
        print("Final cost is {:.3f}".format(T[-1, -1]))
        return img_map[-1, -1], None, None


class OpenCL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = None

    def loadProgram(self, filename):
        f = open(filename, 'r')
        f_str = "".join(f.readlines())
        self.program = cl.Program(self.ctx, f_str).build()

    def opencl_energy(self, img):
        mf = cl.mem_flags

        H, W, D = map(np.int32, img.shape)
        img = img.astype(np.float32).reshape(-1)
        res = np.empty(img.shape).astype(np.float32)

        self.img_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
        self.res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, res.nbytes)

        self.program.energy(self.queue, img.shape, None, self.img_buf, self.res_buf, H, W, D)
        cl.enqueue_read_buffer(self.queue, self.res_buf, res).wait()

        res = res.reshape((H, W, D))
        return np.sum(res, axis=2)


os.environ["PYOPENCL_CTX"] = "0:1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

example = OpenCL()
example.loadProgram("get_energy.cl")

if __name__ == "__main__":
    FILE = os.path.join('img', 'nature_512.png')
    image = mpimg.imread(FILE)
    original = np.copy(image)

    print("Original image shape:", image.shape)
    image, eng, path = seam_carve(image, dw=10, dh=0)
    print("\rFinal image shape:", image.shape)

    # plot
    #H, W = original.shape[:2]
    #show_image(1, original, "Original", H, W)
    #show_image(2, eng, "Energy plot", H, W, path)
    #show_image(3, image, "Seam carving", H, W)
    #plt.show()