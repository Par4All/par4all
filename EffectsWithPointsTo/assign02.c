typedef struct {int entier; float reel;} s_t;

s_t make_s(int i, float x)
{
  s_t * p;
  extern char * malloc(int);

  p = (s_t *) malloc(sizeof(s_t));
  p->entier = i;
  p->reel = x;
  return *p;
}

void assign02()
{
  s_t s;
  int j;

  s = make_s(1, 2.);
  s.entier = 4;

  /* invalid lvalue in assignment */
  /* make_s(1, 2.).entier = 4; */

  /* But this is OK in rhs */
  j = make_s(1, 2.).entier;
}
