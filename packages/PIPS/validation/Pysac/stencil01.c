/* from the paper stVec at PACT2011 */
#include <stdlib.h>
#include <stdio.h>
void stencil01(int n, float a[n], float b[1+n]) {
    for(int i=0;i<n;i++)
        a[i]=(b[i]+b[i+1])*.5;
}

int main(int argc, char * argv[]) {
    int n = argc>1?atoi(argv[1]):103;
    float a[n],b[1+n];
    for(int i=0;i<=n;i++)
        b[i]=i;
    stencil01(n,a,b);
    float csum = 0.;
    for(int i=0;i<=n;i++)
        csum+=a[i];
    printf("%f\n",csum);
    return 0;
}
