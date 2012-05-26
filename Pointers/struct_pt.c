#include<stdlib.h>
int main()
{
  typedef struct my_struct {
    int i;
    int *p;
  } my_str;
  my_str *m, *n;
  int j;
  j = 2;
  m = (my_str *) malloc(sizeof(my_str));
  m->i = 1;
  m->p = &j;
  n = m;

  return 0;
}
