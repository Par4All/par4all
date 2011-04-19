#include <stdio.h>
int main() {
   struct one {
    int *first;
   } x;
  int y[10];

  x.first = &y[5];


  return 0;
}
