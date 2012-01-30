
void scilab_rt_mean_i3_d0(int in00, int in01, int in02, int matrixin0[in00][in01][in02],
     double* scalarout0)
{
  int i;
  int j;
  int k;

  int val0 = 0;
  for (i = 0; i < in00; ++i) {
    for (j = 0; j < in01; ++j) {
      for (k = 0; k < in02; ++k) {
        val0 += matrixin0[i][j][k];
      }
    }
  }

  *scalarout0 = val0;

}
