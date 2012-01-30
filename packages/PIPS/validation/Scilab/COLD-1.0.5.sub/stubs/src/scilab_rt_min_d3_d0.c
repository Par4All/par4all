
void scilab_rt_min_d3_d0(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02],
        double *out0)
{

  int i, j, k;
  double val1 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      for (k = 0; k < sin02; ++k) {
        val1 += in0[i][j][k];
      }
    }
  }
  *out0 = val1;

}

