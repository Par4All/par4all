#include <stdio.h>
int a(int i) {
    int j[2]={2,2},k[2],l;
    j[0]=0;
    l=0;
    if(i)
        k[0]=1;
    else
        l=i;
    load(k,j);
    return k[0]+k[1];
}
int main(int argc, char *argv[]) {
    printf("%d\n",a(argc));
    return 0;
}

