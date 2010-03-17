/* Short version of io_intrinsics.c, designed to debug the
   prettyprinter which is bothered by ICO C99 specific functions. */

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

int main(char *fmt1, ...)
{
  FILE* fp;
  size_t n, nr;
  char * fmt2;
  char * i_name;
  int i, r, c, max;
  fpos_t * fp_pos, pos;
  long int fp_pos_indic;
  va_list vl;


  (void) fscanf(fp, fmt2,i_name, &i);
  va_start(vl, fmt1);
  (void) vfscanf(fp, fmt1, vl);
  va_end(vl);
  va_start(vl, fmt1);
  (void) vfprintf(fp, fmt1, vl);
  va_end(vl);
}

