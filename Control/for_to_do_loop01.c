#include <stdlib.h>
short accumulate(size_t n, short a[n],short seed)
{
    size_t i;
    for(i=0;i<n;i++)
        seed+=a[i];
    return seed;
}
