#include <stdio.h>
int a(int i) {
    int j[2]={2,2},k[2],l,m;
    j[0]=0;
    for(m=0;m<10;m++) {
        for(l=0;l<10;l++) {
            store(k,j);
            i++;
        }
    }
    return k[0]+k[1];
}
int main(int argc, char *argv[]) {
    printf("%d\n",a(argc));
    return 0;
}

