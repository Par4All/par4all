/* Check impact of variable length arrays on flatten code*/

#include <stdio.h>

void vla01() {
  int i;
  {
    int i;
    scanf("%d", &i);
    int a[i];
    a[0] = 1;
  }
}
