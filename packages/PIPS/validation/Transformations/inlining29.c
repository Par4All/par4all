#include "array.h"
int main(int argc, char*argv[])
{
    int i,n = atoi(argv[1]);
    float sum;
    farray a = farray_new(n);
    for(i=0;i<n;i++)
        farray_set(a,i,(float)i);
    sum=0;
    for(i=0;i<n;i++)
        sum+=farray_get(a,i);
    printf("%f\n",sum/n);
    return 0;
}
