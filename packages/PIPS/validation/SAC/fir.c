#include <stdio.h>
#include <stdlib.h>

void FIRFilter( size_t n, float in[n], float out[n], float kernel[n],
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
    size_t i,n=argc==1?10:atoi(argv[1]);
    float in[n],out[n],kernel[n];
    for(i=0;i<n;i++){
        in[i]=i;
        kernel[i]=n-i;
    }
    FIRFilter(n,in,out,kernel,i/4);
    for(i=0;i<n;i++){
        printf("%d.",(int)out[i]);
    }
    return 0;
}
