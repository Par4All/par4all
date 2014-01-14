#include <stdio.h>

// Check that effect on static variables are preserved, at least
// when they escape
//
// If the line j = j + c is removed, c = c + 1 becomes useless...
int COUNT(int J) {
  static int C = 0;
  J = J + C;
  C = C + 1;
  return J;
}

int LOST_COUNT(int J) {
  static int C = 0;
  C = C + 1;
  return J;
}

int LOST_COUNT_2(int J) {
  static int C = 0;
  J = J + C;
  C = C + 1;
  return 0;
}

int main() {
  int J=0;
  J=COUNT(J);
  J=LOST_COUNT(J);
  J=LOST_COUNT_2(J);
  printf("%d\n",J);
}


