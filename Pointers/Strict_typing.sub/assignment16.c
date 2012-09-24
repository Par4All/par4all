/* Check assigment of arrays to pointers
 *
 * Bug: r_cell_reference_to_type()...
 */

#include <stdio.h>

struct array {
  int a[10][10];
} s;

int assignment16(struct array s)
{
  int *q=s.a[5];

  q+=3;

  return *q;
}

int main()
{
  struct array s;
  int i;
  int j;
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      s.a[i][j] = 10*i+j;
  int k = assignment16(s);
  printf("%d\n", k);
  return 0;
}
