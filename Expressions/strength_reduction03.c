#include <stdio.h>
main() {
    int i=0,*a=malloc(10*sizeof(int)),b[10];
    if(!a) return 1;
    else {
        int *c=&b[0];
        for(i=0;i<10;i++)
            *(a+i)=2+*(c+i);
        printf("%d",*(a+2));
        return 0;
    }
}
