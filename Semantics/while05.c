/* Side effects in while condition
 */

int while05()
{
  int i, n;

  i = 0;
  n = 10;

  while(--n) {
    i++;
  }
  return i;
}
