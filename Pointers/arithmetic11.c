/* Check pointer arithmetic
 *
 */

#include <stdio.h>

int main()
{
  int n = 4, m = 3;
  int a[n][m];
  int (*p)[m] = a;
  p += 1; 
 
  return 0;
}
