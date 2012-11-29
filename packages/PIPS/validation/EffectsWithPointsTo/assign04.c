/* To be out of Emami's patterns
 *
 * assign04 segfaults...
 */

void assign04()
{
  int *** p;
  int ** q;
  int * r;
  int i;

  r = &i;
  //q = &r;
  p = &q;
  **p = r;
  ***p = 0;
  return;
}

void foo()
{
  assign04();
  return;
}
