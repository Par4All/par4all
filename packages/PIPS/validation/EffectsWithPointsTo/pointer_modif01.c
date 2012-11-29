/* This code is bugged because p is uninitialized...
 *
 * Is this test case useful?
 */

int main()
{
  int *p, *q;

  *p = 1;

  q = p;
  *q = 2;
  return 0;
}
