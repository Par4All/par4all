#include <stdlib.h>

typedef struct {
    int re,im;
} icplx_t;

int main(int argc, char *argv[]) {
    icplx_t i,k ;
    i = (icplx_t){0,1};
    k = (icplx_t){argc,i.im};
    return 0;
}

