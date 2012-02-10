typedef struct {
    double im,re;
} complex_t;
void sain(complex_t *t) {
    t->re=t->im=1;
}
double pain() {
    complex_t t ;
holy:sain(&t);
     return t.re+t.im;
}

#include <stdio.h>
int main() {
    printf("%ld\n",pain());
    return 0;
}
