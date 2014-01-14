/* Check naming of stubs according to fields used: extension of struct15.c
 *
 * FI: this code is all bugged since it uses indeterminate
 * pointers. It segfault at execution. However, the current analysis
 * is performed with the formal context and not with the actual
 * context for main, and it should not detect the issue.
 */

#include <stdio.h>

struct two {
  float ** fifth;
};

struct one {
  int **first;
  char **second;
  float **third;
  struct two fourth;
} x;

int main() {
  int y[10];
  char z[10];
  float v[10];

  *(x.first) = &y[5];
  *(x.second) = &z[6];
  *(x.third) = &v[7];
  *(x.fourth.fifth) = &v[8];

  return 0;
}
