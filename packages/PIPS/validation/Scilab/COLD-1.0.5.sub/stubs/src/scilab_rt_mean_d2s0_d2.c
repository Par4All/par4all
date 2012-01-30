
void scilab_rt_mean_d2s0_d2(int in00, int in01, double matrixin0[in00][in01], 
    char* scalarin0,
    int out00, int out01, double matrixout0[out00][out01])
{
  int i;
  int j;

  double val0 = 0;
  if (scalarin0) {
    for (i = 0; i < in00; ++i) {
      for (j = 0; j < in01; ++j) {
        val0 += matrixin0[i][j];
      }
    }

    for (i = 0; i < out00; ++i) {
      for (j = 0; j < out01; ++j) {
        matrixout0[i][j] = val0;
      }
    }
  }
}
