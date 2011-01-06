#include <stdio.h>
int a(int i) {
    int j[2]={2,2},k[2],l;
    j[0]=0;
    if(i) {
        store(k,j);
        l=0;
    }
    return k[0]+k[1];
}
int main(int argc, char *argv[]) {
    printf("%d\n",a(argc));
    return 0;
}

