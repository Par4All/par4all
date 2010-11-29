#include <stdio.h>
main() {
    int i=0,*a=malloc(10*sizeof(int));
    if(!a) return 1;
    for(i=0;i<10;i+=2) {
        *(a+i)=3;
        *(a+i+1)=4;
    }
    printf("%d",*(a+2));
    return 0;
}
