
int scilab_rt_or_d3_(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02])
{
  int i;
  int j;
  int k;

  double val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      for (k = 0; k < sin02; ++k) {
        val0 += in0[i][j][k];
      }
    }
  }

  return val0;
}

