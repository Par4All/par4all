/* To debug the handling of effetcs in declarations
 *
 * FI: I have lost the effects in initialization expression (1 May 2013)
 */

void declaration01(int *p)
{
  int *q = p;
  // q = p;
  return *q;
}
