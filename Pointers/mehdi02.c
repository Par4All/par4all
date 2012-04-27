/* Bug in memory management */

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

  p = (struct p_s*)malloc(sizeof(struct p_s));
  p->q = x;

  return 0;
}
