#include <stdio.h>
int a(int i) {
    int j[2]={1,2},k[2];
    while (i--) {
        load(k,j);
        k[0]=i;
        store(j,k);
    }
    return j[1];
}
int main() {
    printf("%d\n",a(2));
    return 0;
}

