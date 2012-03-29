/* Check pointer arithmetic
 *
 */

void arithmetic07(int a[10])
{
  int * q = &a[9];
  int * p;

  p = q-1;
  return;
}
