
void scilab_rt_mean_i2_d0(int in00, int in01, int matrixin0[in00][in01],
     double* scalarout0)
{
  int i;
  int j;

  int val0 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      val0 += matrixin0[i][j];
    }
  }

  *scalarout0 = val0;

}
