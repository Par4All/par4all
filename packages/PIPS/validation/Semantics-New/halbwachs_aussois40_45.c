/* Nicolas Halbwachs, slide 40 out of 45, Aussois, Dec. 2011 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
{
  return rand()%2;
}

int main()
{
  int v, t, x, d;
  v = t = x = d = 0;

  // The convex enveloppe of the guards is true
  while(1){
    while(x<=4 && d<=9)
      if(alea())
	x++, v++;
      else
	d++, t++;
    //if(x>4)
    while(/*x>4 &&*/ d<=9)
	d++, t++;
    //else if(d>9)
    while(/*d>9 &&*/ x<=4 && (alea() || x>=2))
	    x++, v++;
    while(d==10 && x >= 2 /* && alea() */)
      x = d = 0;
  }
}
