#include <stdio.h>
int a(int i,int j[2], int k[2]) {
    load(k,j);
    store(k,j);
    if(j[0])
        return i;
    i++;
}
int main() {
    int J[2]={1,2},K[2];
    a(2,J,K);   
    printf("%d\n",K[0]+K[1]);
    return 0;
}

