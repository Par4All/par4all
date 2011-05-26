/* Example by Sven showing Laure being more precise than Sven */

#include <stdlib.h>

int alea(void)
{
  return rand() % 2;
}

int skimo01(int x, int y, int n)
{
  int z = 0;
  if(n<2) exit(1);

  while(alea()) {
    x++;
    y += 1-n;
  }
  z +=x+y;
  return z;
}
