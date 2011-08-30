#include <stdio.h>
#include <assert.h>
#define alpha 40
void alphablending(int n,int src0[n][n], int src1[n][n])
{
    unsigned int i,j;
#pragma terapix
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            src0[i][j]=(
                    alpha*src0[i][j]
                    +
                    (100-alpha)*src1[i][j]
                    )/*/100*/;
}

int main(int argc, char * argv[])
{
    int check,i,j,n = argc >1 ? atoi(argv[1]): 200;
    int a[n][n], b[n][n];
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            a[i][j]=1,b[i][j]=1;
    alphablending(n,a,b);
    check=0;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            check+=a[i][j];
    printf("%d\n",check);
    return 0;
}
