
int main()
{
  int k;
  int n = 10;
  int a[n][n][n];
  int b[n][n][n];

// Use to generate a pragma openmp with private(j) ; but j is declared inside the loop !
  for (k = 0; k < 8; ++k) {
    int i,j;
    for (i = 0; i < n; ++i) {
      for (j = 0; j < n; ++j) {
        b[k][i][j] = a[k][i][j] + 42;
      }
    }
  }
  return 0;
}

