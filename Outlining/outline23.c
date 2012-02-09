#include <stdio.h>
static void do_assign(int b[1], int i) {
    b[i]=i;
}
void assign(int *b, int i) {
    do_assign(b+i,i);
}

float ing(int i) {
    int b[100];
here:if(1) {
        printf("%d\n",i);
        for(i=0;i<100;i++) {
            assign(b,i);
        }
    }
}
