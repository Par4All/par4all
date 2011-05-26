#include <stdio.h>
int main(void)
{
    int j=1, k=2, l=3;
    j = j * (k - l) * k;
    k = (j+l) + k ;
    l = l - k  - j;
    printf("%d|%d|%d\n",j,k,l);
    return 0;
}

