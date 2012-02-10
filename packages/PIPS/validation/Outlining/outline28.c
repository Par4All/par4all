#include <stdio.h>
enum {
    NO,
    YES
};
typedef enum { PEPSI, COCA } coca;

static void do_assign(int b[1], int i) {
    b[i]=NO;
}
void assign(int* b, int i) {
    coca cola = PEPSI ;
    do_assign(&cola,i);
}

float ing(int i) {
    int b[100];
here:if(1) {
         assign(b,i);
    }
}
