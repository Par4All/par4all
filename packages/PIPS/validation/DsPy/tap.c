/* tap.c - i-th tap of circular delay-line buffer */

double tap(D, w, p, i)                    /* usage: si = tap(D, w, p, i); */
double *w, *p;                            /* \(p\) passed by value */
int D, i;                                 /* \(i=0,1,\dotsc, D\) */
{
       return w[(p - w + i) % (D + 1)];
}
