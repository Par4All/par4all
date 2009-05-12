/* Explicit boolean in test condition with side effect
 */

int if06()
{
  int i, n;

  i = 0;
  //n = 10;

  if(n++>0) {
    i++;
  }
  else {
    i--;
  }

  return i;
}
