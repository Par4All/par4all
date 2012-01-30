// The goal: investigate the impact of convex hull on not included functions

// The two functions from store to convex array regions are very
// different and their convex hull does imply large coefficients.

// This is a 2-D version of convex_hull01.

int main()
{
  int i, k;
  int N = 100;
  double A[N][N];

  for(i=0;i<N;i++) {
    for(k=0;k<N;k++) {
      if(1) {
	A[i][k] = 1.;
	A[N/2][N/2] = 0.;
      }
    }
  }

  return 0;
}
