#include <stdio.h>
int a(int i,int j[2], int k[2]) {
    int l;
    l=0;
    load(k,j);
    j[0]=0;
    l=i;
}
int main() {
    int J[2]={1,2},K[2];
    a(2,J,K);   
    printf("%d\n",K[0]+K[1]);
    return 0;
}

