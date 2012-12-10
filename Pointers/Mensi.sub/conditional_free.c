#include<stdbool.h>
#include<stdlib.h>

void conditional_free(int *p, bool c1, bool c2, bool c3) {
  if(c1) p = (int *) malloc(sizeof(int));
  if(c2) free(p);
  if(c3) p = (int *) malloc(sizeof(int));
  return;
}
