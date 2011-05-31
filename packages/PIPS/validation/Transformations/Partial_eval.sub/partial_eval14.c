/* How do we handle C divison differently from Fortran? They seem to
   be identical, although the modulo is defined differently

   So we might have a problem with a == (a/b)*b + a %b, but do not.
*/

#include <stdio.h>

int check_mod_div(int a, int b)
{
  return a == (a/b)*b + a %b;
}

int main()
{
  int i;
  i = (3)/(2);
  printf("pos-pos 3/2=%d (must be 1) and check_mod_div %d\n", i, check_mod_div(3,2));
  i = (-3)/(2);
  printf("neg pos (-3)/2=%d (must be -1) and check_mod_div %d\n", i, check_mod_div(-3,2));
  i = (3)/(-2);
  printf("pos neg 3/(-2)=%d (must be -1) and check_mod_div %d\n", i, check_mod_div(3,-2));
  i = (-3)/(-2);
  printf("neg neg (-3)/(-2)=%d (must be 1) and check_mod_div %d\n", i, check_mod_div(-3,-2));
  return i;
}
