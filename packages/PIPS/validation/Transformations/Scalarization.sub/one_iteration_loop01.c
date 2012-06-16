// in foo the scalarization phase with property
// SCALARIZATION_KEEP_PERFECT_PARALLEL_LOOP_NESTS=TRUE,
// the expected result is either that out[j] is not scalarized
// in the function body or that the loop is no more declared as parallel
// or that the loop is removed: it has only one iteration
// since it is always called with size=1.

#include <stdio.h>

typedef struct {

  float   re;
  float   im;
} Cplfloat;


void foo(int size, Cplfloat in[size], float out[size]) {
 int j;

 for (j = 0;  j < size;  j++) {
  out[j] = in[j].re*in[j].re + in[j].im*in[j].im;
 }

}

int main()
{
  Cplfloat IN1[8][193][32];
  float OUT1[8][193][32];
  int i0;
  int i1;
  int i2;

  for (i0 = 0; i0 < 8; i0++) {
    for (i1 = 0; i1 < 193; i1++) {
      for (i2 = 0; i2 < 32; i2++) {

	IN1[i0][i1][i2].re = (float) i0;
	IN1[i0][i1][i2].im = (float) i0;
      }
    }
  }

  for (i0 = 0; i0 < 8; i0++) {
    for (i1 = 0; i1 < 193; i1++) {
      for (i2 = 0; i2 < 32; i2++) {

	foo(1, (Cplfloat(*))&IN1[i0][i1][i2], (float(*))&OUT1[i0][i1][i2]);

      }
    }
  }

  for (i0 = 0; i0 < 8; i0++) {
    for (i1 = 0; i1 < 193; i1++) {
      for (i2 = 0; i2 < 32; i2++) {

	printf("OUT1[%d][%d][%d] = %f\n", i0, i1, i2, OUT1[i0][i1][i2]);
      }
    }
  }
  return 0;
}
