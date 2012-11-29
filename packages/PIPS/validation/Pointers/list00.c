#include "list.src/list.c"

int main(void)
{
  list l;
  l = nil;
  l = list_cons(1.0, nil);
  l = list_cons(2.0, l);
  l = list_cons(3.0, l);
  int n = list_len(l);
  list_free(l);
  return 0;
}
