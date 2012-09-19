/* Check assigment of arrays to pointers
 *
 * Bug: r_cell_reference_to_type()...
 */

void foo(int * pi)
{
  *pi = 123;
}

int assignment11()
{
  struct {
    int a[10][20];
  } s;
  int *q=s.a[5];

  foo(s.a[5]);

  return *q;
}
