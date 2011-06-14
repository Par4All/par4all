

int main() {
  int M = rand();
  int N = rand();
  int i,j,k;
  double alpha = 1;
  double C[M][N];

  // j should be replace by its bound in the inner loop expression
  for (j = 0; j < N; j++) {
    for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha;
    }
  }
}
