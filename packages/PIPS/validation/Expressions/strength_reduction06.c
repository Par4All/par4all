#include <stdio.h>
main() {
    int i=0,*a=malloc(10*sizeof(int)),*b;
    if(!a) return 1;
    b=a;
    for(i=8;i>0;i-=2) {
        *(a+i)=*(b+i+1);
    }
    printf("%d",*(a+2));
    return 0;
}
