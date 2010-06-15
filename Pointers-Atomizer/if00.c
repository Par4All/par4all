#include <stdio.h>
int main() {
  int **x, *x1, x2,*y, y1;
   x2 = 0;
   y1 = 1;
   x1  = &x2;
 
   if(true)
     {
     
      x = &x1;
      y = &y1;
     }
   else
      *x = &y;

  return 0;
}
