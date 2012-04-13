#include<stdlib.h>
//#include<stdio.h>

int ptr_to_field()
{
  typedef struct {
    int *q;
    int *p;
  }my_str;
  
  my_str *m, *n;
  int i=0, j=1;
  m = (my_str*) malloc(sizeof(my_str));
  n = (my_str*) malloc(sizeof(my_str));
  m->q = &i;
  m->p = &j;
  n->q = m->p;
  n->p = m->q;
  
  return 0;
}
