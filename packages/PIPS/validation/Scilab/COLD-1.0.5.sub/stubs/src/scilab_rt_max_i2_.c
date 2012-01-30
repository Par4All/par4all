
int scilab_rt_max_i2_(int si00, int si01, int in0[si00][si01])
{

  int i, j;

  int val1 = 0;
  for (i = 0; i < si00; ++i) {
    for (j = 0; j < si01; ++j) {
      val1 += in0[i][j];
    }
  }
  return val1;
}

