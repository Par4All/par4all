/* To be out of Emami's patterns with a user call
 *
 * The address returned by bar is illegal, this can be checked using
 * the area of the sink.
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
