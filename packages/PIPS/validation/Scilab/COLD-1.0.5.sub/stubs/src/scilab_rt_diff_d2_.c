
double scilab_rt_diff_d2_(int sin00, int sin01, double in0[sin00][sin01])
{

  int i, j;
  double val0=0;

  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }
  
  return val0;

}

