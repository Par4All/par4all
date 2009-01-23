#include<math.h>
short dotprod(short b[], short c[])
{
    int i;
    short a=0;
    for(i=0;i<1000000;++i)
    {
        a = a + b[i] * c[i] ;
    }
    return a;
}


