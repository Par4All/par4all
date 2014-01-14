/* pi is written and ends up pointing to nowhere, but it does not
 * matter for the caller because pi is a copy of aipp.
 *
 * Note the disapperance of if(1) wrt formal_parameter01.c
 */

int formal_parameter01b(int **pi)
{
  int ** q;
  int *i;
  int j;

  i = 0;
  q = pi;
  q++; // Incompatible with call site since pi points toward a scalar
  pi = &i;
  *pi = &j;
  q[j] = &j; // Incompatible with call site unless j==0, see previous comment

  return 0;
}

int main()
{
  int i, *ip, **aipp;
  ip = &i;
  aipp = &ip;
  i = formal_parameter01b(aipp);
  return 0;
}
