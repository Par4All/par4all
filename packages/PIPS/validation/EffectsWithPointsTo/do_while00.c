#include <stdio.h>
int main() {
  int *a, *b, c, d, i;
   c = 0;
   d = 1;
   i = 2;
   b = &c;
 
   do{
     i = i+1;
     a= b;
     b = &d;
   }while(i<5); 

  return 0;
}
