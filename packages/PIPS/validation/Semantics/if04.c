/* Explicit boolean in test condition
 */

int if04()
{
  int i, n;

  i = 0;
  //n = 10;

  if(n>0) {
    i++;
    n--;
  }
  else {
    i--;
    n++;
  }

  return i;
}
