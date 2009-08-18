#include <stdlib.h>

void alphablending(size_t n, float src1[n], float src2[n], float result[n], float alpha)
{
    size_t i;
    for(i=0;i<n;i++)
        result[i]=alpha*src1[i]+(1-alpha)*src2[i];
}
