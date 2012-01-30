
void scilab_rt_min_i2_i0(int sin00, int sin01, int in0[sin00][sin01],
    int *out0)
{
  int i;
  int j;

  int val1 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val1 += in0[i][j];
    }
  }
  *out0 = val1;
}

