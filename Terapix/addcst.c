#include <stdio.h>
void runner(int n,int out[n][n], int in[n][n],int mask) {
    int i,j;
#pragma terapix
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            out[i][j]=in[i][j]+mask;
}

int main(int argc, char ** argv) {
    int check,i,j,n = argc >1 ? atoi(argv[1]): 200;
    int out[n][n], in[n][n],mask[1]={2};
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            in[i][j]=1;
    runner(n,out,in,mask[0]);
    check=0;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            check+=out[i][j];
    printf("%d\n",check);
    return 0;
}


