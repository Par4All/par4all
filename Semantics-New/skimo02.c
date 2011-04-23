/* Example by Sven showing Laure being more precise than Sven
 *
 * Same as skimo01, but with n checked at a higher level so as to get
 * an interesting transformer for skimo02.
 *
 * Still not available at the function level because parameters are
 * passed by value.
 */

#include <stdlib.h>
#include <stdio.h>

int alea(void)
{
  return rand() % 2;
}

void skimo02(int x, int y, int n)
{
  while(alea()) {
    x++;
    y += 1-n;
  }
  return;
}

int main()
{
  int x, y, n;
  scanf("%d",&n);
  if(n<2) exit(1);
  skimo02(x, y, n);
}
