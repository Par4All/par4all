/* pi is written and ends up pointing to nowhere, but it does not
 * matter for the caller because pi is a copy of aipp.
 *
 * Note the disapperance of if(1) wrt formal_parameter01.c. The
 * execution error is now detected.
 */

int formal_parameter01a(int **pi)
{
  int ** q;
  int *i;
  int j;

  i = 0;
  q = pi;
  q++; // Incompatible with call site since pi points toward a scalar
  pi = &i;
  *pi = &j;
  *q = &j; // Incompatible with call site, see previous comment

  return 0;
}

int main()
{
  int i, *ip, **aipp;
  ip = &i;
  aipp = &ip;
  i = formal_parameter01a(aipp);
  return 0;
}
