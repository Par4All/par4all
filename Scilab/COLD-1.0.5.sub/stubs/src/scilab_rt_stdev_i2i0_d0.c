
void scilab_rt_stdev_i2i0_d0(int sin00, int sin01, int in0[sin00][sin01], int in1, double *out0)
{
  int i,j;
  int val0=0;

  if (in1) {
    for (i = 0; i < sin00; ++i) {
      for (j = 0; j < sin01; ++j) {
        val0 += in0[i][j];
      }
    }
  }

  *out0 = val0;


}
