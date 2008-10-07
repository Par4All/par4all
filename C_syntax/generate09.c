/* Same as generate08, but both structures are named. */

struct first {
  int a;
  double b;
};

typedef struct second {
  struct first c;
  char d;
} sem_t;

void generate09()
{
  sem_t t;
  (void) func(&t);
}
