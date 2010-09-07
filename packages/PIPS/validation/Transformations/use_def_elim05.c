#include <stdlib.h>

typedef struct data_st {
  int x;
} data_t;

data_t * data_create(void)
{
  data_t * d = malloc(sizeof(data_t));
  d->x = 0;
  return d;
}

void data_free(data_t * d)
{
  free(d);
}

int data_add(data_t * o, data_t * i, data_t * j)
{
  o->x = i->x + j->x;
  return 0;
}

int data_or(data_t * o, data_t * i, data_t * j)
{
  o->x = i->x | j->x;
  return 0;
}

int data_stuff(data_t * o, data_t * i, data_t * j)
{
  data_t * t = data_create();
  data_add(t, i, j);
  data_or(o, t, j);
  data_free(t);
  return 0;
}
