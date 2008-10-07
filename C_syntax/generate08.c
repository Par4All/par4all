struct first {
  int a;
  double b;
};

typedef struct {
  struct first c;
  char d;
} sem_t;

void generate08()
{
  sem_t t;
  //extern int fun(sem_t *);
  (void) func(&t);
}
