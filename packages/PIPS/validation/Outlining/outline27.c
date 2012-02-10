#include <stdio.h>
typedef struct { int * pepsi ; } coca;

static void do_assign(int b[1], int i) {
    b[i]=i;
}
void assign(int* b, int i) {
    coca cola = { b } ;
    do_assign(cola.pepsi,i);
}

float ing(int i) {
    int b[100];
here:if(1) {
         assign(b,i);
    }
}
