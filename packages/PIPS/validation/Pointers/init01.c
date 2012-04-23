// Check initialization of points-to in declaration statements

struct cons_t {
  double value;
  struct cons_t * next;
};

typedef struct cons_t * list;

void init01()
{
  int *p;
  list q;
  struct cons_t r;
  return;
}
