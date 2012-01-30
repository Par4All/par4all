
double scilab_rt_mul_d2d2_(int sin00, int sin01, double in0[sin00][sin01],
    int sin10, int sin11, double in1[sin10][sin11])
{
  int i;
  int j;

  double val0 = 0;
  double val1 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }
  for (i = 0; i < sin10; ++i) {
    for (j = 0; j < sin11; ++j) {
      val1 += in1[i][j];
    }
  }

  return val0 + val1;
}

