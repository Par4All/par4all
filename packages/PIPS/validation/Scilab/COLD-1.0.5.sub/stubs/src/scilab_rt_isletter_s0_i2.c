
void  scilab_rt_isletter_s0_i2(char* s, int n, int m, int res[n][m])
{
  int i,j;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      res[i][j] = *s;
    }
  }

}
