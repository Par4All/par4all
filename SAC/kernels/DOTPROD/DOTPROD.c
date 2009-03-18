#include<math.h>
float dotprod(float b[1000000], float c[1000000])
{
    int i;
    float a=0;
    for(i=0;i<1000000;++i)
    {
        a = a + b[i] * c[i] ;
    }
    return a;
}


