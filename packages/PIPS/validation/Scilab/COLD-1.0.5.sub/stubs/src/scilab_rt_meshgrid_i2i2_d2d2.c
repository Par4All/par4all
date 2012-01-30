
void scilab_rt_meshgrid_i2i2_d2d2(int nx, int mx, int x[nx][mx],
    int ny, int my, int y[ny][my],
    int nz, int mz, double z[nz][mz],
    int nw, int mw, double w[nz][mz])
{

  int i, j;
  int val0=0, val1=0;

  for (i = 0; i < nx; i++)
    for (j = 0; j < mx; j++)
      val0 += x[i][j];

  for (i = 0; i < ny; i++)
    for (j = 0; j < my; j++)
      val1 += y[i][j];

  for (i = 0; i < nz; i++)
    for (j = 0; j < mz; j++)
      z[i][j] = val0 + val1;

  for (i = 0; i < nw; i++)
    for (j = 0; j < mw; j++)
      w[i][j] = val0 + val1;

}



