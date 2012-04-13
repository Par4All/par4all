/* Check assigment of structures
 *
 */

int assignment08(int a[10])
{
  struct foo {int * a; int * b;} x, y;
  int i, j;

  x.a=&i;
  x.b=&j;
  y = x;

  return 0;
}
