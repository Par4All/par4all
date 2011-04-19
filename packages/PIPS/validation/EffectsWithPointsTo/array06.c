/* From Ronan Keryell (see trac ticket #150) */

#include <stdio.h>

int main() {
  int t[100][10][3];
  int *pu;
  
  pu = t[1][2];
  return 0;
}
