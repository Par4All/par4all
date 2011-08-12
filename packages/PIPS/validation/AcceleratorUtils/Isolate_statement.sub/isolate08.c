#include <stdio.h>
#define n 40
#define alpha 40
void alphablending(short src0[n][n], short src1[n][n], short result[n][n])
{
    unsigned int i,j;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            result[i][j]=(
                    alpha*src0[i][j]
                    +
                    (100-alpha)*src1[i][j]
                    )/100;
}

int main(int argc, char * argv[])
{
    short a[n][n],b[n][n],c[n][n];
    int i,j;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            a[i][j]=b[i][j]=i*j;
    alphablending(a,b,c);
    for(i=0;i<n;i++){
        for(j=0;j<n;j++)
            printf("%hd ",c[i][j]);
        puts("\n");
    }
    return 0;
}
