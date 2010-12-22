#include <stdio.h>
int a(int i, int k[2]) {
    int l,j[2]={1,2};
    l=0;
    load(k,j);
    j[0]=0;
    l=i;
}
int main() {
    int J[2]={1,2},K[2]={3,4};
    a(2,K);   
    printf("%d\n",K[0]+K[1]);
    return 0;
}

