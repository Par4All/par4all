/* See how the lattice interact with targets of array elements...
 *
 * simplified version of array16.c to see what the value of q becomes...
 *
 * It shows a bug in the dereferencement of p[*] in the analyzer
 *
 * The value of "ii" is irrelevant for the points-to analysis, but
 * useful to silence gcc.
 */

void array16a()
{
  int a, b;
  int * p[10], *q;
  int ii = 2;
  p[0]=&a;
  p[1]=&b;
  q = p[ii];
  return;
}
