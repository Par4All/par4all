// Check initialization of points-to in declaration statements when a
// struct contains an array of pointers

struct content {
  int flag;
  int * data[10];
};

struct cons_t {
  double value;
  struct content foo;
  struct cons_t * next[10];
};

typedef struct cons_t * list;

void init03x()
{
  struct cons_t a;
  return;
}
