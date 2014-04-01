/* Vivien suggested a non affine bounded domain...
 *
 * Here is a case with 8 points in 2-D, same as rotation02 but with
 * random transitions.
 *
 * The current list-based approach fails although the initial state s0 is
 * in image(bopy) and image(body) = constant U s0, with s0 in body
 *
 * The property SEMANTICS_K_FIX_POINT must be set to 2
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
{
  return rand()%2;
}


void rotation03()
{
  int x = 1, y = 0;

  while(1) {
    if(alea())
      x=2, y=0;
    if(alea())
      x=3, y=1;
    if(alea())
      y=2, x=3;
    if(alea())
      x=2, y=3;
    if(alea())
      x=1, y=3;
    if(alea())
      x=0,y=2;
    if(alea())
      y=1, x=0;
    if(alea())
      x=1,y=0;
  }
}
