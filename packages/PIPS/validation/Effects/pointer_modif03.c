#include <malloc.h>

typedef struct {int entier; float reel;} s_t;

s_t make_s(int i, float x)
{
  s_t * p;

  p = (s_t *) malloc(sizeof(s_t));
  p->entier = i;
  p->reel = x;
  return *p;
}

int main()
{
  s_t s;
  int j;

  s = make_s(1, 2.);
  s.entier = 4;

  j = make_s(1, 2.).entier;

  return j;
}
