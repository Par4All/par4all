#include <stdio.h>

// Check that effect on static variables are preserved, at least
// when they escape
//
// If the line j = j + c is removed, c = c + 1 becomes useless...
int count(int J) {
  static int C = 0;
  J = J + C;
  C = C + 1;
  return J;
}

int lost_count(int J) {
  static int C = 0;
  C = C + 1;
  return J;
}

int main() {
  int J=0;
  J=count(J);
  J=lost_count(J);
  printf("%d\n",J);
}


