/* Implicit boolean in test condition
 */

int if01()
{
  int i, n;

  i = 0;
  //n = 10;

  if(n) {
    i++;
    n--;
  }
  else {
    i--;
    n++;
  }

  return i;
}
