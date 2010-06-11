/* From Ronan Keryell (see trac ticket #150) */

#include <stdio.h>

int main() {
  int t[100][10][3], i;

  int (*p)[3];
  for(i=0; i <3; i++)
    p = t[i];

  return 0;
}
