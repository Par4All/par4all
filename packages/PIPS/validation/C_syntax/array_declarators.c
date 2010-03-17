/* This code is extracted from C ISO standard*/
extern int n;
extern int m;
void fcompat(void)
{
  int a[n][6][m];
  int (*p)[4][n+1];
  int c[n][n][6][m];
  int (*r)[n][n][n+1];
}
