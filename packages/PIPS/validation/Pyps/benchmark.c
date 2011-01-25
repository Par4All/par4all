#include <stdlib.h>
#include <stdio.h>

void run(int n, int a[n], int b[n], int c[n]) {
    for(int i=0;i<n;i++)
        a[i]+=b[i]*c[i];
}
int main(int argc, char ** argv) {
    int n = argc == 1 ? 100 : atoi(argv[1]);
    int c=0;
    int A[n],B[n],C[n];
    for(int i=0;i<n;i++) B[i]=-(C[i]=i);
    for(int j=0;j<n;j++) A[j]=0;
    run(n,A,B,C);
    for(int k=0;k<n;k++) c+=A[k];
    printf("%d",c);
    return 0;
}
