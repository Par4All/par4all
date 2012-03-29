/* Check pointer arithmetic
 *
 */

void arithmetic03(int *p)
{
  int * q;

  q = p;
  // not standard compliant as p does not points towards an array, but
  // gcc -c -Wall -std=c99 does not mention it... I guess it would
  // kill -Werror
  q++;
  return;
}
