#include <stdio.h>
#ifndef __PIPS__
#define MIN(a,b) ((a)>(b)?(a):(b))
#endif
void runner(int n, int img_out[n][n-2], int img[n][n], int mask[3]) {
    int x, y,z;
#pragma terapix
    for (y = 0 ; y < n  ; ++y) {
        for (x = 1; x < n - 1; x++) {
            int val = 0;
            for(z=-1;z<2;z++) {
                val+=img[y][x+z]*mask[1+z];
            }
            img_out[y][x-1]=val;
        }
    }
}

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 200;
    int out[n][n-2], in[n][n], mask[3]={1,1,1};
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in[i][j]=j;
    runner(n,out,in,mask);
    check=0;
    for(i=0;i<n;i++)
        for(j=0;j<n-4;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}



