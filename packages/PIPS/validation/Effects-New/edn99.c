#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#include "edn99_ppm.h"

#ifndef CLK_TCK
#define CLK_TCK      CLOCKS_PER_SEC
#endif


int main(int argc, char *raw_argv[]) {
  FILE *inf, *outf;
  char *in_fn = NULL, out_fn[256] = "";

  // Make the arg list nice.
  char *argv[argc * 2];

  // Open the input.
  if(!in_fn) {
    fprintf(stderr, "%s: missing file name\n", argv[0]);
    return 7;
  }
  inf = fopen(in_fn, "rb");
  if(inf == NULL) {
    fprintf(stderr, "%s: can't open '%s' for reading\n", argv[0], argv[1]);
    return 8;
  }

  // Read the input image.
  ppm_dim dim = get_ppm_dim(inf);
  return 0;
}
