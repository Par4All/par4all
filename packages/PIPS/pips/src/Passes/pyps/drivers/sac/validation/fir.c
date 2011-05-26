#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

void fir( size_t n, float in[n], float out[n], float kernel[n],
        size_t ksize )
{
    size_t i, j;
    for( i = 0; i < n - ksize + 1; ++i )
        for( j = 0; j < ksize; ++j )
            out[ i ] += in[ i + j ] * kernel[ j ];
}

int main(int argc, char * argv[]) {
    size_t i;
    size_t n;
    
    if (argc<3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    if (n<=100)
	    n=100;
    {
        float (*in)[n],(*out)[n],(*kernel)[n];
        in=malloc(sizeof(float)*n);
        out=calloc(n,sizeof(float));
        kernel=malloc(sizeof(float)*n);
        init_data_file(argv[2]);
        init_data_float(in, n);
        init_data_float(kernel, n);
        close_data_file();
        fir(n,*in,*out,*kernel,n/8);
        print_array_float("res", out, n);
        free(out);
        free(in);
        free(kernel);
    }
    return 0;
}
