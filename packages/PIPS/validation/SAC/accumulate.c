/* pips is not good at parsing std headers and unsplit afterward */
typedef unsigned int size_t;
short accumulate(size_t n, short a[n],short seed)
{
    size_t i;
    for(i=0;i<n;i++)
        seed=seed+a[i];
    return seed;
}
