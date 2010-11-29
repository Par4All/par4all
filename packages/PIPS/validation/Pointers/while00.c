#include <stdio.h>
int main() {
  int *a, *b, c, d;
   c = 0;
   d = 1;
   b = &c;
 
   while(c > 0){
     a = b;
     b = &d;
   }

   b = &c;

  return 0;
}
