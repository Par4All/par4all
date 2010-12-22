#include <stdio.h>
int a(int i) {
    int j[2]={1,2},k[2],l;
    j[0]=0;
    l=0;
    store(k,j);
    l=i;
    return k[0]+k[1];
}
int main() {
    printf("%d\n",a(2));
    return 0;
}

