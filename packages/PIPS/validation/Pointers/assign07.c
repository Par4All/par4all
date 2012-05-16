/* Same as assign04.c, but different property in tpips: undefined can
 * be dereferenced.
 */

void assign07()
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
  assign07();
}
