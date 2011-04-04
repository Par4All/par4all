/* wrap2.c - circular wrap of pointer offset q, relative to array w */

void wrap2(int M, int *q)
{
       if (*q > M)  
              *q -= M + 1;          /* when \(*q=M+1\), it wraps around to \(*q=0\) */

       if (*q < 0)  
              *q += M + 1;          /* when \(*q=-1\), it wraps around to \(*q=M\) */
}
