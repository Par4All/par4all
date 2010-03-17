#include <stdio.h>
int main() {
   struct one {
    int *first;
   } *x, x1;
   int y[10], z[5];
   x1.first = &z[1];
   (*x).first = x1.first;
   (*x).first = &y[5];


  return 0;
}
