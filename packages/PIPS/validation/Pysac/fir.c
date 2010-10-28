#include <stdio.h>
#include <stdlib.h>

void fir( size_t n, float in[n], float out[n], float kernel[n],
        size_t ksize )
{
    size_t i, j;
    for( i = 0; i < n - ksize + 1; ++i )
    {
        out[ i ] = 0.0;
        for( j = 0; j < ksize; ++j )
            out[ i ] += in[ i + j ] * kernel[ j ];
    }
}

int main(int argc, char * argv[]) {
    size_t i;
    size_t n=argc==1?10000:atoi(argv[1]);
    float (*in)[n],(*out)[n],(*kernel)[n];
    in=malloc(sizeof(float)*n);
    out=malloc(sizeof(float)*n);
    kernel=malloc(sizeof(float)*n);
    for(i=0;i<n;i++){
        (*in)[i]=i;
        (*kernel)[i]=n-i;
    }
    fir(n,*in,*out,*kernel,n/8);
    for(i=0;i<n;i++){
        printf("%d.",(int)(*out)[i]);
    }
    free(out);
    free(in);
    free(kernel);
    return 0;
}
