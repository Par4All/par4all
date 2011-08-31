#include <stdio.h>
#ifndef __PIPS__
#define MIN(a,b) ((a)>(b)?(a):(b))
#endif
void runner(int n, int img_out[n][n-4], int img[n][n]) {
    int x, y;
#pragma terapix
    for (y = 0 ; y < n  ; ++y) {
        for (x = 2; x < n - 2; x++) {
            img_out[y][x-2] = MIN(MIN(MIN(MIN(img[y][x-2], img[y][x-1]), img[y][x]), img[y][x+1]), img[y][x+2]);  
        }
    }
}

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 200;
    int out[n][n-4], in[n][n];
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in[i][j]=j;
    runner(n,out,in);
    check=0;
    for(i=0;i<n;i++)
        for(j=0;j<n-4;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}



