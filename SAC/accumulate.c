#include <stdio.h>

short accumulate(unsigned int n, short a[n],short seed)
{
    unsigned int i;
    for(i=0;i<n;i++)
        seed=seed+a[i];
    return seed;
}
int main()
{
    unsigned int n =12,i;
    short a[n];
    short b;
    for(i=0;i<n;i++)a[i]=i;
    b=accumulate(n,a,3);
    printf("%d",b);
    return 0;
}
