#include <stdio.h>
int a(int i) {
    int j[2]={2,2},k[2],l;
    j[0]=0;
    for(l=0;l<10;l++) {
        load(k,j);
        i++;
        j[0]=1;
    }
    return k[0]+k[1];
}
int main(int argc, char *argv[]) {
    printf("%d\n",a(argc));
    return 0;
}

