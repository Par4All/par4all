#include <stdlib.h>

typedef struct {
    int re,im;
} icplx_t;

int main(int argc, char *argv[]) {
    icplx_t i ;
    i = (icplx_t){0,1};
    int one = ((icplx_t){0,1}).im;
    return 0;
}

