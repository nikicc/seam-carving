__kernel void part1(__global float* img, __global float* results)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int z = get_global_id(2);

    int index = x + 320*y + 320*512*z;

    results[index] = img[index];
}
