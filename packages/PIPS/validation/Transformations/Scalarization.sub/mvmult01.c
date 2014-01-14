/* Example by Ramakrishna Upadrasta */

#include <stdio.h>

void mvmult01(int n, double a[n], double b[n], double m[n][n])
{
  int i, j;

  for(i=0; i<n; i++) {
    a[i]= 0;
    for(j=0; j<n; j++)
      a[i] += m[i][j]*b[j];
  }
}
				 
int main()
{
  int n;
  double a[n], b[n], m[n][n];
  mvmult01(n, a, b, m);
  printf("%g\n", a[0]);
}
