// Bug: unfeasible loop exit condition

#include <stdio.h> /* stderr */

#define BLOCKSIZE 16
#define MATSIZE 128

int main()
{
  int i, k, j, cpi, cpj;		// indexes used in loops
  float l[MATSIZE*MATSIZE];

  // Number of blocks
  int n = MATSIZE / BLOCKSIZE;
  float tmp[BLOCKSIZE * BLOCKSIZE];
  float _tmp1[BLOCKSIZE * BLOCKSIZE];

  cpi = BLOCKSIZE;
  cpj = BLOCKSIZE;

  for (cpi = 0; cpi < BLOCKSIZE; cpi++) {
    for (cpj = cpi+1; cpj < BLOCKSIZE; cpj++) {
      unsigned idx1 = cpi * BLOCKSIZE + cpj;
      unsigned idx2 = cpj * BLOCKSIZE + cpi;
      _tmp1[idx2] = tmp[idx1];
    }
    //printf("cpj=%d\n", cpi);
  }

  printf("3 cpi=%d\n", cpi);

  return 0;
}
