// example taken from the paper
// Pack Transposition: Enhancing Superword Level Parallelism Exploitation
// C. Tenllado, L. Pinuel, M. Prieto, F. Catthoor
//
#include <stdio.h>
#define N 5
void pack_transposition(double a[N][N])
{
    int i,j;
    for(i=0;i<N-1;i++)
        for(j=0;j<N-3;j++)
            a[i][j+1] = a[i][j]+a[i][j+2];
}
int main()
{
    double a[N][N];
    int i,j;
    for(i=0;i<N;i++)
    for(j=0;j<N;j++)
        a[i][j]=i*j;
    pack_transposition(a);
    for(i=0;i<N;i++)
    for(j=0;j<N;j++)
        printf("-%f-",a[i][j]);
    return 0;
}


