/* To be out of Emami's patterns with a user call
 *
 * FI: this is not a good example beause "bar" it flawed.  We should
 * at least use a global variable instead of "return &j".
 *
 */

int * bar(int j)
{
  return &j;
}

void assign05()
{
  int * r;
  int i;

  r = bar(i);
  i = 1;
  *r = 0;
}

void foo()
{
  assign05();
}
