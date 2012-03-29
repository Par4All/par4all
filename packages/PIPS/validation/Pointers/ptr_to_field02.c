/* Bug with rhs in assignment n->q = m->q */

#include<stdlib.h>
#include<stdio.h>

int ptr_to_field()
{
  typedef struct {
    int *q;
  }my_str;
  
  my_str *m, *n;
  int i=0, j=1;
  m = (my_str*) malloc(sizeof(my_str));
  n = (my_str*) malloc(sizeof(my_str));
  m->q = &i;
  n->q = m->q;
  
  return 0;
}
