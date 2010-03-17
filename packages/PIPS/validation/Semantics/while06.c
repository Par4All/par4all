/* Implicit boolean while condition
 */

int while06()
{
  int i, n;

  i = 0;
  n = 10;

  while(n) {
    i++;
    n--;
  }
  return i;
}
