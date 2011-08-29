

int main() {
  int M = rand();
  int N = rand();
  int i,j,k;
  double alpha = 1;
  double C[M][N];

  // j must be replaced by its bounds in the outermost annotation
  for (j = 0; j < N; j++) {
    for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha;
    }
  }
  // j must be replaced by its bounds in the outermost annotation
  for (j = 0; j < N; j++) {
    for (k = N-j; k < N; k++) {
        C[k][j] += alpha;
    }
  }

}
