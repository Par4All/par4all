
#include <complex.h>

void scilab_rt_pow_z2i0_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int in1,
    int sout00, int sout01, double complex out0[sout00][sout01])
{
  int i;
  int j;

  double complex val0 = 0;
  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }

  val0 += in1;

  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
        out0[i][j] = val0;
    }
  }

}
