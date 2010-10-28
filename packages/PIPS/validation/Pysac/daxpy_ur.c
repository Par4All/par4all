#include <stdio.h>
#include <stdlib.h>


void daxpy_ur(int n,float da,float dx[n],float dy[n])
{
    int i,m;
    m = n % 4;
    if ( m != 0)
    {
        for (i = 0; i < m; i++)
            dy[i] = dy[i] + da*dx[i];
        if (n < 4)
            return;
    }
    for (i = m; i < n; i = i + 4)
    {
        dy[i] = dy[i] + da*dx[i];
        dy[i+1] = dy[i+1] + da*dx[i+1];
        dy[i+2] = dy[i+2] + da*dx[i+2];
        dy[i+3] = dy[i+3] + da*dx[i+3];
    }
}

int main(int argc, char * argv[])
{
    int n = argc==1?10:atoi(argv[1]);
    float (*dx)[n],(*dy)[n];
    int i;
    dx=malloc(sizeof(float)*n);
    dy=malloc(sizeof(float)*n);
    for(i=0;i<n;i++)
        (*dx)[i]=i;
    daxpy_ur(n,42.,*dx,*dy);
    for(i=0;i<n;i++)
        printf("%f",(*dy)[i]);
    free(dx);
    free(dy);
    return 0;
}

