#include <stdio.h>

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 200;
    int out[n][n], in0[n][n], in1[n][n];
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in0[i][j]=1,in1[i][j]=1;
#pragma terapix
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            out[i][j]=in0[i][j]-in1[i][j];
    check=0;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}


