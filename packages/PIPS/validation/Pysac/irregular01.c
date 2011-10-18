#include <stdio.h>
#include <stdlib.h>
void irregular01(int n, int a[n], int b[n]) {
    for(int i=0;i<n;i++) {
        a[i]=a[b[i]+1];
    }
}
int main(int argc, char * argv[]) {
    int n = 1 + (argc>1 ? atoi(argv[1]) : 103);
    int a[n], b[n];
    for(int i=0;i<n;i++)
        b[i]=i/2;
    irregular01(n,a,b);
    int sum = 0.;
    for(int i=0;i<n;i++)
        sum+=a[i];
    printf("%d\n",sum);
    return 0;
}
