/* Example of Linear Relation Analysis given by Nicolas Halbwachs in his
   great tutorial at Aussois, 9/12/2010

   http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/programme.html
 */

#include <stdio.h>

int main(int argc, char *argv[]) {
  int x = 0;
  int y = 0;
  char b;

  while(x <= 100) {
    b = getchar();
    if (b)
      x = x + 2;
    else {
      x = x + 1;
      y = y + 1;
    }
  }
  /* After widening: 0 <= y <= x , x >= 101 */
  return x;
}
