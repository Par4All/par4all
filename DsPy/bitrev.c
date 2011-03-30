/* bitrev.c - bit reverse of a B-bit integer n */

#define two(x)       (1 << (x))                  /* \(2\sp{x}\) by left-shifting */

int bitrev(int n, int B)
{
       int m, r;

       for (r=0, m=B-1; m>=0; m--)
          if ((n >> m) == 1) {                   /* if \(2\sp{m}\) term is present, then */
             r += two(B-1-m);                    /* add \(2\sp{B-1-m}\) to \(r\), and */
             n -= two(m);                        /* subtract \(2\sp{m}\) from \(n\) */
             }

       return(r);
}
