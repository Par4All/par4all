/* Example by Sven showing Laure being more precise than Sven
 *
 * skimo01.c modified to grab the impact of the loop on alea() via
 * variable z. See transformer of skimo03
 */

#include <stdlib.h>

int alea(void)
{
  return rand() % 2;
}

int skimo03(int x, int y, int n)
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
