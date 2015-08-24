#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define ABS(x) ((x)<0 ? -(x) : (x))

__kernel void part1(__global float* img, __global float* results, int height, int width)
{
    unsigned int ind = get_global_id(0);

    unsigned int x = ((ind/4)%height);
    unsigned int y = (ind/(4*height));
    unsigned int z = (ind%4);

    int back = (x*width*4) + (y*4) + z;

    int w = x;
    int h = y;

    int w_l = MAX(0, w-1);
    int w_r = MIN(w+1, width-1);
    int h_u = MAX(0, h-1);
    int h_d = MIN(h+1, height-1);

    float der = 0;
    int back_left = (w_l*width*4) + (y*4) + z;
    int back_right = (w_r*width*4) + (y*4) + z;
    der += ABS(img[back_left] - img[back_right]);

    int back_up = (x*width*4) + (h_u*4) + z;
    int back_down = (x*width*4) + (h_d*4) + z;
    der += ABS(img[back_up] - img[back_down]);

    results[back] = der;
}
