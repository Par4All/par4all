/* test case to illustrate the need to update points-to by using
   eval_cell_with_points_to() */
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
 struct q_s* b;
 struct q_s *u;
 struct q_s *x;
 struct q_s **z;
 int a = 0;
 u =(struct q_s*) malloc(sizeof(struct q_s));
 b = (struct q_s*)malloc(sizeof(struct q_s));
 p = (struct p_s*)malloc(sizeof(struct p_s));
 x = b;
 p->q = x;



 z = &(p->q);
 p->q = b;
 z = &u;

 p->q->r = a;
 return 0;
}
