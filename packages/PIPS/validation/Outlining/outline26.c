#include <stdio.h>
typedef int hint;
typedef hint *hintptr;

static void do_assign(int b[1], int i) {
    b[i]=i;
}
void assign(hintptr b, int i) {
    do_assign(b+i,i);
}

float ing(int i) {
    int b[100];
here:if(1) {
         assign(b,i);
    }
}
