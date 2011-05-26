/* Speed counter

   Example of Linear Relation Analysis given by Nicolas Halbwachs in his
   great tutorial at Aussois, 9/12/2010

   http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/programme.html
 */

#include <stdio.h>

int main(int argc, char *argv[]) {
  int t, d, s;

  t = d = s = 0;

  while(1) {
    // Just to display infos on this block
    if (1) {
      if (s <= 3) {
	s++;
	d++;
      }
      {
	t++;
	s = 0;
      }
    }
  }
  /* Unreachable */
  return 0;
}
