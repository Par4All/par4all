#include <stdio.h>
int main() {
  int *a, *b, c, d, i;
   c = 0;
   d = 1;
   b = &c;
 
   for(i = 1; i<5; i++){
     c = 5;
     a = b;
     b = &d;
   }

  return 0;
}
