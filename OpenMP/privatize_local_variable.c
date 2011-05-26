
int main()
{
  int k;
  int n = 10;
  int a[n][n];

// Use to generate a pragma openmp with private(j) ; but j is declared inside the loop !
  for (k = 0; k < 8; ++k) {
    int i,j;
    int b[n][n];
    for (i = 0; i < n; ++i) {
      for (j = 0; j < n; ++j) {
        b[i][j] = a[i][j] + 42;
      }
    }
  }
  return 0;
}

