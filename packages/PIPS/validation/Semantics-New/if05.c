/* Purpose: check side effect in condition, here, j=0
 */

int if05()
{
  int i, j, k;

  if(0<=j && j<=2) {
    if(j=0)
      k = 0;
    else
      k = 4;
  }

  return k;
}
