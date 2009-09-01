extern void * malloc(int);
extern void free(void *);

void suppress_dead_code01(int *x)
{
  char * p;

  /* Not dead code because of memory read-write effects, even if
     masked because the implementation is thread-safe. */
  p = malloc(10);
  free(p);

  /* Not dead code because of control effect */
  while(1) {
    ;
  }

  /* Not dead code because of memory read-write effects, even if
     masked because the implementation is thread-safe, but dead
     because the previous loop never terminates. */
  p = malloc(10);
  free(p);
}
