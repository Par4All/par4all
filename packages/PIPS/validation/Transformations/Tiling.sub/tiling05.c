/* Check for tiling in C */

#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

/*
void heat(int n, float (*a)[n][n])
{
  int i, j;

  for(i=1; i<n; i++)
    for(j=1; j<n-1; j++)
      (*a)[i][j] = 0.25*((*a)[i-1][j-1]+2*(*a)[i-1][j]+(*a)[i-1][j+1]);
}
*/

/* For code generation, a bit of PIPS run-time... that is conflicting
   with Fortran intrinsics */
 /*
int pips_min(int i, ...)
{
  int m = i;
  int j;
  va_list pa;
  va_start(pa, i);
  j = va_arg(pa, int);
  va_end(pa);
  m = m>j? j : m;
  return m;
}
int max(int i, ...)
{
  int m = i;
  return i<j? j : i;
  return m;
}

double average(int count, ...)
{
    va_list ap;
    int j;
    double tot = 0;
    va_start(ap, count); //Requires the last fixed parameter (to get the address)
    for(j=0; j<count-1; j++)
        tot+=va_arg(ap, double); //Requires the type to cast to. Increments ap to the next argument.
    va_end(ap);
    return tot/count;
}
*/

void tiling05(int n, float a[n][n])
{
  int i, j;

 l100:  for(i=1; i<n; i++)
    for(j=1; j<n-1; j++)
      a[i][j] = 0.25*(a[i-1][j-1]+2*a[i-1][j]+a[i-1][j+1]);
}

int main()
{
  int n = 10;
  int i, j;
  float a[n][n];

  for(i=0; i<n; i++)
    for(j=0; j<n; j++)
      if(i==0 && j>0 && j<n-1)
	a[i][j] = (float) j;
      else
	a[i][j] = 0.;

  for(i=0; i<1; i++)
    for(j=1; j<n-1; j++)
      printf("a[%d][%d]=%f\n", i, j, a[i][j]);

  tiling05(n, a);

  for(i=n-1; i<n; i++)
    for(j=1; j<n-1; j++)
      printf("a[%d][%d]=%f\n", i, j, a[i][j]);
  return 0;
}
