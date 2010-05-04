#include <stdio.h>
void main()
{
    int al,lo;
    for(lo=0;lo<10;lo++)
        al=MAX(al,lo);
    printf("%d\n",al);
}
