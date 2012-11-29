/* Same as mehdi02.c but different tpips script: here
 * context-sensitive is set to false.
 *
 * p->q cannot be evaluated because q is not initialized.
 */

#include<stdio.h>
#include<stdlib.h>

int main()
{
  struct p_s{
    struct q_s *q;
  };

  struct q_s{
    int r;
  };
  struct p_s* p;
  struct q_s *x;

  p = (struct p_s*) malloc(sizeof(struct p_s));
  p->q = x;

  return 0;
}
