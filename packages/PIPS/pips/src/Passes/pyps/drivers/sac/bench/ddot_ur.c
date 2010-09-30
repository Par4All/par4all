#include <stdlib.h>
#include <stdio.h>

float ddot_ur(int n,float dx[n], float dy[n])
{
    int i,m;
    float dtemp=0;
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dtemp = dtemp + dx[i]*dy[i];
        if (n < 5)
            return(dtemp);
    }
    for (i = m; i < n; i = i + 5)
    {
        dtemp = dtemp + dx[i]*dy[i] +
            dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2] +
            dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4];
    }

    return dtemp;
}

int main(int argc, char ** argv)
{
    int i,n=argc==1?200:atoi(argv[1]);
    float a, (*b)[n], (*c)[n];
    b = malloc(n * sizeof(float));
    c = malloc(n * sizeof(float));
    for(i=0;i<n;i++)
    {
        (*b)[i]=(float)i;
        (*c)[i]=(float)i;
    }
    a=ddot_ur(n,*b, *c);
    printf("%f",a);
    free(b);
    free(c);
    return 0;
}

