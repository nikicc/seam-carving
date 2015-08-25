import numpy as np

H = 5
W = 8
D = 4

x = np.random.rand(H, W, D)
xl = np.copy(x).reshape(-1)
print(x.shape)
print(xl.shape)

for h in range(H):
    for w in range(W):
        for d in range(D):
            # check index calculation
            index = h*W*D + w*D + d

            print("h={}, w={}, d={}\t".format(h, w, d), end='')
            if xl[index] == x[h, w, d]:
                print("OK\t", end='')
            else:
                print("**** NOK ***\t", end='')

            # check back calculation
            back_d = index % D
            if back_d == d:
                print("OK\t", end='')
            else:
                print("**** NOK ***\t", end='')

            back_h = index // (W*D)
            if back_h == h:
                print("OK\t", end='')
            else:
                print("**** NOK ***\t", end='')

            back_w = (index // D) % W
            if back_w == w:
                print("OK\t")
            else:
                print("**** NOK ***\t")
