#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

/*
        GAUSSIAN_NOISE_ REDUCE
        apply 5x5 Gaussian convolution filter, shrinks the image by 4 pixels in each direction, using Gaussian filter found here:
        http://en.wikipedia.org/wiki/Canny_edge_detector
        code dl from http://code.google.com/p/fast-edge
*/
void gaussian_noise_reduce(int w, int h, float img_in[w*h], float img_out[w*h])
{
    int  x, y, max_x, max_y;
    max_x = w - 2;
    max_y = w * (h - 2);
    for (y = w * 2; y < max_y; y += w) {
        for (x = 2; x < max_x; x++) {
            img_out[x + y] = (2 * img_in[x + y - 2 - w - w] + 
                    4 * img_in[x + y - 1 - w - w] + 
                    5 * img_in[x + y - w - w] + 
                    4 * img_in[x + y + 1 - w - w] + 
                    2 * img_in[x + y + 2 - w - w] + 
                    4 * img_in[x + y - 2 - w] + 
                    9 * img_in[x + y - 1 - w] + 
                    12 * img_in[x + y - w] + 
                    9 * img_in[x + y + 1 - w] + 
                    4 * img_in[x + y + 2 - w] + 
                    5 * img_in[x + y - 2] + 
                    12 * img_in[x + y - 1] + 
                    15 * img_in[x + y] + 
                    12 * img_in[x + y + 1] + 
                    5 * img_in[x + y + 2] + 
                    4 * img_in[x + y - 2 + w] + 
                    9 * img_in[x + y - 1 + w] + 
                    12 * img_in[x + y + w] + 
                    9 * img_in[x + y + 1 + w] + 
                    4 * img_in[x + y + 2 + w] + 
                    2 * img_in[x + y - 2 + w + w] + 
                    4 * img_in[x + y - 1 + w + w] + 
                    5 * img_in[x + y + w + w] + 
                    4 * img_in[x + y + 1 + w + w] + 
                    2 * img_in[x + y + 2 + w + w]) / 159;
        }
    }
}

int main(int argc, char **argv) {
    int n;
    float (*in)[n*n], (*out)[n*n];
    int i;
    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    in=malloc(n*n);
    out=malloc(n*n);
    init_data_file(argv[2]);
    init_data_float(in, n*n);
    close_data_file();
    for(i=0;i<n*n;i++) {
        (*out)[i]=0;
    }
    gaussian_noise_reduce(n,n,*in,*out);
    print_array_float(out, n*n);
    free(in);free(out);
    return 0;
}



