/* FI: One of the gaz burner examples in Laure Gonnord's PhD
 * dissertation
 *
 * Restructured by hand. Because of the nested loops, the transformers
 * must be recomputed. I'm not sure transformer lists are useful
 * although the internal loops are potentially nasty because of calls
 * to alea(): they may do nothing
 *
 * The while() {while(); while();} trick is used to avoid a test in
 * the middle of a while loop.
 *
 * For Vivien Maisonneuve and Corinne Ancourt
 *
 * The problem with Vivien restructuring may be limited to one node
 * which has two possible transitions. This node should be
 * restructured with the above trick about replacing an if by a pair
 * of while statements.
 *
 * Note: many overflows after transformer refinement
 */

#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  return rand()%1;
}


int main()
{
  float z;
  int u,l,t,v;

  u=l=t=v=0;
  while(1) {
    u = v = 0;
    while(u<60) {
      if(1) {
	while(u<60 && v<=9 && alea())
	  l++, v++, u++, t++;
	while(u<60 && alea())
	  t++, u++;
	if (6*l<= t+50) printf("property verified\n");
	else printf("error\n");
      }
    }
    if (6*l<= t+50) printf("property verified\n");
    else printf("error\n");
  }
  if (6*l<= t+50) printf("property verified\n");
  else printf("error\n");
}
