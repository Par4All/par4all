// The goal: investigate the impact of convex hull on not included functions

// The two functions from store to convex array regions are very
// different and their convex hull does imply large coefficients.

// This is simplified from chopped_corner24: a one D version,
// but with the same difficulty, same as in the initial case from COLD...

// It is so simplified that the summation after the convex hull
// succeeds without overflows even if N is increased by several order of
// magnitudes and if the constant reference is shifted.

int main()
{
  int k;
  int N = 100;
  double A[N];

  for(k=0;k<N;k++) {
    if(1) {
      A[k] = 1.;
      A[N/2] = 0.;
    }
  }

  return 0;
}
