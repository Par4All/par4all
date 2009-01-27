#include<math.h>
short dotprod(short b[1000000], short c[1000000])
{
    int i;
    short a=0;
    for(i=0;i<1000000;++i)
    {
        a = a + b[i] * c[i] ;
    }
    return a;
}


