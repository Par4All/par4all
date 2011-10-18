// Engineered for Ronan Keryell: although the timing functions have an
// RW effect on the hidden clock, the loop is distributed because the
// statement a[i]=0; can be executed in a parallel loop and because
// there are no dependencies between a general computation and a time
// effect

#include "time.h"

#define N 100

int main()
{
  float a[N];
  double b[N];
  clockid_t c;
  time_t t1, t2;
  int i;

  for(i=0;i<N;i++) {
    t1 = time(&t1);
    a[i] = 0.;
    t2 = time(&t2);
    b[i] = difftime(t2, t1);
  }
  return 0;
}
