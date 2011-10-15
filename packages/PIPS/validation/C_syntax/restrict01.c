/* Check prettyprint of restrict */

/* non standard version: void restrict01(int * __restrict p, int * __restrict q);*/
/* C99 version: void restrict01(int * restrict p, int * restrict q);*/

void restrict01(int * restrict p, int * restrict q);

void restrict01a(const int * restrict p, int * const q);

void restrict01b(__const int * __restrict p, int * __const q);

float* __restrict__ xx;
