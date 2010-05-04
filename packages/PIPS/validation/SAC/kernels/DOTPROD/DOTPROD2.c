#include <stdio.h>
float inner_loop(int _i, float *_b, float *_c)
{
    return _b[_i] + _c[_i];
}

float dotprod(int n, float b[n], float c[n])
{
    float a;
    int i;
    for(i=0;i<n;++i)
    {
        a+=inner_loop(i,b,c);
    }


    return a;
}
int main(int argc, char **argv)
{
    int n=atoi(argv[1]);
    {
        float B[n],C[n];
        return (int)dotprod(n,B,C);
    }
}


