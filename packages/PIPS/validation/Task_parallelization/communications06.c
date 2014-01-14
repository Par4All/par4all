/* Check parametric communications
 *
 * Same as communications03.c but here
 * the array size is static, the loop bounds are static and defined by the preprocessor. 
 * 
 * The communication pattern depends on the relative value of bi and
 * bj. Some communications are going to turn empty at run time.
 */

#include <stdio.h>
#include <assert.h>


int main(int argc, char *argv[])
{
  int size;
  int a[100];
  int i, bi, j;
  scanf("%d", &size);
  scanf("%d", &bi);
  for(i=0; i<bi; i++)
    a[i] = i;
  for(i=bi; i<size; i++)
    a[i] = i;
  for(j=0; j<size; j++)
    printf("%d\n", a[j]);
  return 0;
}
