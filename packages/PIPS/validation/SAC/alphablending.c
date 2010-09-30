#include <stdio.h>

void alphablending(unsigned int n, float src1[n], float src2[n], float result[n], float alpha)
{
    unsigned int i;
    for(i=0;i<n;i++)
        result[i]=alpha*src1[i]+(1-alpha)*src2[i];
}
void beta(int n) {
    float src1[n], src2[n], result[n];
    alphablending(n,src1,src2,result,.2f);
    { int i;
        for(i=0;i<n;i++)
            printf("%f",result[i]);
    }
}
