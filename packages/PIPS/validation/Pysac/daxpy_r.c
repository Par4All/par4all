#include <stdio.h>
#include <stdlib.h>


void daxpy_r(int n,float da,float dx[n],float dy[n])
{
    int i;
    /* code for both increments equal to 1 */

    for (i = 0;i < n; i++)
        dy[i] = dy[i] + da*dx[i];
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
    daxpy_r(n,42.,*dx,*dy);
    for(i=0;i<n;i++)
        printf("%f",(*dy)[i]);
    free(dx);
    free(dy);
    return 0;
}

