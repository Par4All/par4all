/* Check parametric communications
 *
 * This is a very modified version of communications01. The arrays are
 * 2-D and they are partitioned in the same way, so communications are
 * not useful, at least before the print. Four tasks should be defined.
 *
 * The array size is dynamic, the loop bounds are dynamic, the
 * communication pattern does not depend on the relative value of bi and
 * bj. Some communications are going to turn empty statically.
 */

#include <stdio.h>
#include <assert.h>

int main()
{
  int size;
  scanf("%d", &size);
  int a[size][size], b[size][size];
  int i, bi, j, bj;
  scanf("%d", &bi);
  scanf("%d", &bj);
  assert(0<=bi && bi<size && 0<=bj && bj<size);

  /* Initialization of a with four tasks */
  for(i=0; i<bi; i++)
    for(j=0; j<bj; j++)
      a[i][j] = i*size+j;
  for(i=0; i<bi; i++)
    for(j=bj; j<size; j++)
      a[i][j] = i*size+j;
  for(i=bi; i<size; i++)
    for(j=0; j<bj; j++)
      a[i][j] = i*size+j;
  for(i=bi; i<size; i++)
    for(j=bj; j<size; j++)
      a[i][j] = i*size+j;

  /* Computation of b with four tasks perfectly aligned on the
     previous four ones */
  for(i=0; i<bi; i++)
    for(j=0; j<bj; j++)
      b[i][j] = 2*a[i][j];
  for(i=0; i<bi; i++)
    for(j=bj; j<size; j++)
      b[i][j] = 2*a[i][j];
  for(i=bi; i<size; i++)
    for(j=0; j<bj; j++)
      b[i][j] = 2*a[i][j];
  for(i=bi; i<size; i++)
    for(j=bj; j<size; j++)
      b[i][j] = 2*a[i][j];

  for(i=0; i<size; i++)
    for(j=0; j<size; j++)
      printf("%d\n", b[i][j]);

  return 0;
}
