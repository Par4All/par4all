/* Triply nested loops for Vivien Maisonneuve's PhD
 *
 * CFG version of nested03
 */

#include <stdio.h>

int main()
{
  int i=0, j, k, l=0, n=10;

  /*
  for(i=0;i<n;i++)
  for(j=0;j<n;j++)
  for(k=0;k<n;k++)
    l++;
  */

 si: if(i>=n) goto se;
  j = 0;
 sj: if(j<n) { k=0; goto sk; }
  i++;
  goto si;
 sk: if(k<n) {k++, l++; goto sk;}
  j++;
  goto sj;

 se:
  printf("l=%d\n", l);
  return 0;
}
