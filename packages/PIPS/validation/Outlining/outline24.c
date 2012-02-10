#include <stdio.h>
typedef int *intptr;

static void do_assign(int b[1], int i) {
    b[i]=i;
}
void assign(intptr b, int i) {
    do_assign(b+i,i);
}

float ing(int i) {
    int b[100];
here:if(1) {
        intptr p;
    }
}
