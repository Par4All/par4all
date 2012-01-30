/* Example for ACCA */

#include <stdlib.h>
#include <stdio.h>

/*  */
void compute(int p, int n, float a[n][n]) {
  float t[n];
  int k = n;
  int i, j;
  for(i=0;i<n;i++) {
    k--;
    t[k] = a[i][i];
    for(j=0;j<n;j++)
      if(p>0)
	a[i][j] /= t[k];
  }
}

/* reads input, initializes parameter p, invokes computation and writes output  */
int main() {
  float a[100][100];
  int n = read_input(100, a);
  int p = init_parameter(n, a);
  compute(p, n, a);
  write_output(n, a);
}

/* computes p as a function of a (simplified here) */
int init_parameter(int n, float a[n][n]) {
  int p;
  while(p<=0) p++;
  return p;
}

/* returns the input data and the effective size*/
int read_input(int n, float a[n][n]) {
  if(n>100) exit(1);
  scanf("%f", &a[0][0]);
  return n;
}

void write_output(int n, float a[n][n]) {
  printf("%f", a[0][0]);
}


