import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter
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
    plt.figure(num)
    plt.clf()
    plt.imshow(img)
    if path is not None:
        plt.plot([i[1] for i in path], [i[0] for i in path], 'r-')
    plt.title(title)
    plt.xlim([0, width])
    plt.ylim([height, 0])


def get_energy_image(img):
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


def remove_one_seam(img, trace, horizontal):
    """
    Remove the path from the image.
    :param img: Image
    :param path: Path as a list of (y, x) coordinates.
    :param vertical: Whether we remove vertical or horizontal seam.
    :return: Smaller image.
    """
    if horizontal:    # dirty hack, since reshaping does only work this way, otherwise the image is corrupted
        img = img.swapaxes(0, 1)
        trace = [(x, y) for y, x in trace]

    height, width, depth = img.shape
    mask = np.ones(img.shape, dtype=bool)

    # create the mask
    for h, w in trace:
        mask[h, w, :] = False

    # remove the pixels on the seam
    img = img[mask].reshape(height, width-1, depth)

    # flip back for horizontal
    if horizontal:
        img = img.swapaxes(0, 1)
    return img


FILE = os.path.join('img', 'nature_256.png')
image = mpimg.imread(FILE)
original = np.copy(image)
print("Original image shape:", image.shape)

N = 1
for i in range(N):
    print('\rSeam {}/{}'.format(i, N), end='')
    if 1:
        eng = get_energy_image(image)
        path = find_seam_vertical(eng)
        image = remove_one_seam(image, path, horizontal=False)
    else:
        eng = get_energy_image(image)
        path = find_seam_horizontal(eng)
        image = remove_one_seam(image, path, horizontal=True)

print("\rFinal image shape:", image.shape)

# plot
H, W = original.shape[:2]
show_image(1, original, "Original", H, W)
show_image(2, eng, "Energy plot", H, W, path)
show_image(3, image, "Seam carving", H, W)
plt.show()
