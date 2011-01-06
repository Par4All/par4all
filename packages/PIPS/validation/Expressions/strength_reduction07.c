#include <stdio.h>
main() {
    int i=0,*a=malloc(10*sizeof(int)),*b;
    if(!a) return 1;
    b=a;
    for(i=1;i<9;i++) {
        *(a+i-1)=1;
    }
    printf("%d",*(a+2));
    return 0;
}
