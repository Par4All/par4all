#include <stdio.h>
int a(int i,int j[2], int k[2]) {
    store(k,j);
    for(i=0;i<10;i++) {
        store(j,j);
    }
}
int main() {
    int J[2]={1,2},K[2];
    a(2,J,K);   
    printf("%d\n",K[0]+K[1]);
    return 0;
}

