#include <stdio.h>
#include <assert.h>
#ifndef __PIPS__
#define MIN(a,b) ((a)>(b)?(a):(b))
#endif
void runner(int n, int img_out[n-4][n-4], int img[n][n], int kernel[3][3]) {
    int x, y,ki,kj;
#pragma terapix
    for (x = 2; x < n-2 ; x++) {
        for (y = 2 ; y < n -2 ; ++y) {
            int v = 0;
            for(ki = 0; ki<3; ki++)
                for(kj = 0; kj<3; kj++)
                    v += img[x+ki-1][y+kj-1]* kernel[ki][kj];
            img_out[x-2][y-2] = v;
        }
    }
}

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 400;
    int out[n-4][n-4], in[n][n];
    int kernel[3][3] = { {1,1,1},{1,1,1},{1,1,1} };

    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in[i][j]=j;
    runner(n,out,in,kernel);
    check=0;
    for(i=0;i<n-4;i++)
        for(j=0;j<n-4;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}



