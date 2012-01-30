
void scilab_rt_sum_i3s0_i3(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02], char* in1,
    int sout00, int sout01, int sout02, int out0[sout00][sout01][sout02])
{
  int i;
  int j;
  int k;

  int val0 = 0;

  if (in1) {
    for (i = 0; i < sin00; ++i) {
      for (j = 0; j < sin01; ++j) {
        for (k = 0; k < sin02; ++k) {
          val0 += in0[i][j][k];
        }
      }
    }
  }

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      for (k = 0; k < sout02; ++k) {
        out0[i][j][k] = val0;
      }
    }
  }

}
