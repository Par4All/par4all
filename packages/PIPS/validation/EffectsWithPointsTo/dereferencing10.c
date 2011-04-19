#include <stdlib.h>
#include <stdio.h>

int main()
{
  int ****p, ***p_1, **p_2, *p_3;
  int i, j;
  int ***q_1, **q_2, *q_3;
  i =0;
  j = 1;
  p_3 = &i;
  q_3 = &j;
  p_2 = &p_3;
  q_2 = &q_3;
  p_1 = &p_2;
  q_1 = &q_2;
  p = &p_1;
  ***p = q_3;




  return 0;
}
