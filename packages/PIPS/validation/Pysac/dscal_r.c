#include <stdio.h>
#include <stdlib.h>


void dscal_r(int n,float da,float dx[n])
{
    int i;
    for (i = 0; i < n; i++)
        dx[i] = da*dx[i];
}

int main(int argc, char * argv[])
{
    int n = argc==1?10:atoi(argv[1]);
    float dx[n];
    int i;
    for(i=0;i<n;i++)
        dx[i]=i;
    dscal_r(n,42.,dx);
    for(i=0;i<n;i++)
        printf("%f",dx[i]);
    return 0;
}

