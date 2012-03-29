/* Check pointer arithmetic
 *
 */

void arithmetic08(int a[10])
{
  int * q = a;
  int * p;

  p = q+=1;
  return;
}
