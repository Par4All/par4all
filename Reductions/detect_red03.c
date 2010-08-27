#include <stdio.h>
int main(void)
{
    int j=1, k=2, l=3;
    j = j * (k - j) * (l -k);
    k = l * k - (l - j) * k;
    printf("%d|%d|%d\n",j,k,l);
    return 0;
}

