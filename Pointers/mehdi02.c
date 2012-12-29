/* Bug in memory management
 *
 * p->q cannot be evaluated because q is not initialized: no p is
 * initialized and q is a field. The assignment is OK, but the value
 * assigned is undefined. This aso is OK with the C standard.
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

  p = (struct p_s *) malloc(sizeof(struct p_s));
  p->q = x;

  return 0;
}
