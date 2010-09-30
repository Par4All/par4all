#include <stdio.h>
#include <stdlib.h>


void dscal_ur(int n,float da,float dx[n])
{
    int i,m;
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dx[i] = da*dx[i];
        if (n < 5)
            return;
    }
    for (i = m; i < n; i = i + 5)
    {
        dx[i] = da*dx[i];
        dx[i+1] = da*dx[i+1];
        dx[i+2] = da*dx[i+2];
        dx[i+3] = da*dx[i+3];
        dx[i+4] = da*dx[i+4];
    }
}

int main(int argc, char * argv[])
{
    int n = argc==1?10:atoi(argv[1]);
    float dx[n];
    int i;
    for(i=0;i<n;i++)
        dx[i]=i;
    dscal_ur(n,42.,dx);
    for(i=0;i<n;i++)
        printf("%f",dx[i]);
    return 0;
}

