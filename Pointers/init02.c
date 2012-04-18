// Check initialization of points-to in declaration statements when a
// struct contains a struct

struct content {
  int flag;
  int * data;
};

struct cons_t {
  double value;
  struct content foo;
  struct cons_t * next;
};

typedef struct cons_t * list;

void init02()
{
  int *p;
  list q;
  struct cons_t r;
  return;
}
