/* Impact of conditions on points-to
 *
 * Check the use of pointer comparisons to remove useless arcs
 *
 * Opposite of if08.c
 *
 * Slighlty changed because of Pierre Jouvelot's idea about forgetting
 * the assignment of undefined values... Good for spotting errors, not
 * so good when the analysis result is compared to the result of an
 * execution.
 */

#include <stdio.h>

int main() {
  int *p, *q, *r, i, j;

  p = &i;

  if(i>0) q = &i;

  if(p!=q) {
    // This assignment cannot be executed according to C standard
    // because the value of q is unknown
    r = q;
    i++; // to get points-to information for the true branch
  }
  else {
    r = &i;
    i++; // to get points-to information for the false branch
}
  
  return 0;
}
