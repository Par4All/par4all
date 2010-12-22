#include <stdio.h>
int a(int i) {
    int j[2]={2,2},k[2],l=i;
    j[0]=0;
    while(i-->=0) {
        store(k,j);
    }
    return k[0]+k[1];
}
int main(int argc, char *argv[]) {
    printf("%d\n",a(argc));
    return 0;
}

