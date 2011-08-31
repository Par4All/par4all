#include <stdio.h>
void a(int i,int j[2], int k[2]) {
    while(i-->=0) {
    load(k,j);
    k[0]++;
    store(j,k);
    }
}
int main() {
    int j[2]={1,2},k[2];
    a(0,j,k);
    a(1,j,k);
    printf("%d\n",k[1]);
    return 0;
}

