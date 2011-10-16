// Make sure that forloops that cannot be converted into a do loop are
// not static control. Here both i and j are kinds of loop indices of
// the external loop.

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/time.h>


double A[10][10];
double C[10][10];

int main()
{
  int i, j, k, N;
  k=0;
  for (i=0, j=0; i<N, j<N; i++,j+=i) {
    for (k=0;k<N;k++)
      {
	A[i][j] = C[i][j];
      }
  }
  return 0;
}
