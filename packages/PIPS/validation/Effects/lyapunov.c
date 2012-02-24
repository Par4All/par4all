#include <math.h>
#include <stdlib.h>

void lyapunov_finish_c99 (int np1, double yp[np1], double ly2[np1]) {
    ly2[0]=yp[ 0 ];
}

void lyapunov_finish (int np1, int np2, double *ly2, double *yp_1d) {
    lyapunov_finish_c99 (np1, *(double(*)[np1])yp_1d, *(double(*)[np1])ly2);
    free(yp_1d); // works well without the free
}
