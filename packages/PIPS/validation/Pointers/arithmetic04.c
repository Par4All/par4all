/* Check pointer arithmetic
 *
 */

void arithmetic04(int a[10])
{
  int * q = a;
  int i; // FI: to separate the previous and next statements
  q++;
  q++;
  q++;
  return;
}
