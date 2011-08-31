#include <stdio.h>
#ifndef __PIPS__
#define MIN(a,b) ((a)>(b)?(a):(b))
#endif
void runner(int n, int img_out[n-4][n], int img[n][n]) {
    int x, y, z;
#pragma terapix
    for (y = 2 ; y < n -2 ; ++y) {
        for (x = 0; x < n ; x++) {
            int val = 0;
            for(z=-2;z<3;z++)
                val  = MAX(val,img[y+z][x]);
            img_out[y-2][x]=val;
        }
    }
}

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 400;
    int out[n-4][n], in[n][n];
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in[i][j]=j;
    runner(n,out,in);
    check=0;
    for(i=0;i<n-4;i++)
        for(j=0;j<n;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}



