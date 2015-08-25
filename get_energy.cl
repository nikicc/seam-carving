#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define ABS(x) ((x)<0 ? -(x) : (x))
#define LINEAR(h, w, d, H, W, D) (h*(W*D) + w*(D) + d)


__kernel void energy(__global float* img, __global float* res, int H, int W, int D)
{
    // get liner index
    int index = get_global_id(0);

    // convert index to 3D coordinates
    int d = index % D;
    int h = index / (W*D);
    int w = (index / D) % W;

    // sanity check
    /*
    int back = LINEAR(h, w, d, H, W, D);
    if(back==index){
        res[index] = -42;
    }
    */

    // "border safe" indices
    int wl = MAX(0, w-1);
    int wr = MIN(w+1, W-1);
    int hu = MAX(0, h-1);
    int hd = MIN(h+1, H-1);

    // derivative value
    float der = 0;

    // left-right
    int bl = LINEAR(h, wl, d, H, W, D);
    int br = LINEAR(h, wr, d, H, W, D);
    der += ABS(img[bl] - img[br]);

    // up-down
    int bu = LINEAR(hu, w, d, H, W, D);
    int bd = LINEAR(hd, w, d, H, W, D);
    der += ABS(img[bu] - img[bd]);

    res[index] = der;
}
