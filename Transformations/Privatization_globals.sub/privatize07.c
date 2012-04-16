#include <stdio.h>

# define N 1024

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE x;
DATA_TYPE a[N][N];
DATA_TYPE p[N];



int main(int argc, char* argv[])
{
    int i, j, k;
    int n = N;


    for (j = 1; j < n; ++j)
    {
        x = a[i][j];
        a[j][i] = x * p[i];
    }
    printf("%lf\n",a[2][3]);

    return 0;
}
