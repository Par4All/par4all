#include <stdio.h>
#include <stdlib.h>

float inner_loop(int _i, float *_b, float *_c)
{
    return _b[_i] + _c[_i];
}

float dotprod(int n, float b[n], float c[n])
{
    float a;
    int i;
    a = 0;
    for(i=0;i<n;++i)
    {
        a+=inner_loop(i,b,c);
    }


    return a;
}
int main(int argc, char **argv)
{
    int n;
    int i;
    float *B, *C;
    n = atoi(argv[1]);
    if (posix_memalign((void **) &B, 32, n * sizeof(float)))
	return 3;
    if (posix_memalign((void **) &C, 32, n * sizeof(float)))
	return 3;
    for (i = 0; i < n; i++) {
	B[i] = i;
	C[i] = n - i;
    }
    printf("%f\n", dotprod(n, B, C));
    return 0;
}


