/* Check side effects
 *
 */

void foo(int * p)
{
}

void side_effect01(int a[10])
{
  int * q = a;

  foo(q++);
  return;
}
