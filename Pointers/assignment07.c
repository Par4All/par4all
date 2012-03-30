/* Check assigment of array to pointer
 *
 */

int assignment07(int a[10])
{
  struct foo {int a; int b;} x;
  struct foo * p = &x;
  int * q = &(p->a);
  int * r = &(p->b);

  return q-r;
}
