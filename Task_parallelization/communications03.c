/* Check parametric communications
 *
 * Same as communications01.c but here
 * the array size is static and the loop bounds are static. 
 * 
 * The communication pattern depends on the relative value of bi and
 * bj. Some communications are going to turn empty at run time.
 */

#include <stdio.h>
#include <assert.h>

int main(int argc, char *argv[])
{
  int size;
  size = 100;
  //scanf("%d", &size);
  int a[size], b[size];
  int i, bi, j, bj;
  scanf("%d", &bi);
  scanf("%d", &bj);
  assert(0<=bi && bi<size && 0<=bj && bj<size);
  for(i=0; i<bi; i++)
    a[i] = i;
  for(i=bi; i<size; i++)
    a[i] = i;
  for(j=0; j<bj; j++)
    b[j] = 2*a[j];
  for(j=bj; j<size; j++)
    b[j] = 2*a[j];
  for(j=0; j<size; j++)
    printf("%d\n", b[j]);
}
