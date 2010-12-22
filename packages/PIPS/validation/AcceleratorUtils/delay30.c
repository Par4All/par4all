#include <stdio.h>
int a(int i) {
    int j[2]={1,2},k[2];
    k[0]=0;
    store(k,j);
    i++;
    load(j,k);
    return j[1];
}
int main() {
    printf("%d\n",a(2));
    return 0;
}

