/* Bug found in jpeg source code */

struct test {
  int i;
  int j;
};

static void reset_error_mgr(struct test s)
{
  s.i = 0;
}

struct test * foo (int r) {
  struct test result;
  result.i = 0;
  result.j = r;
  /* Awful, but OK for debugging the PIPS preprocessor */
  return &result;
}


int main () {
  struct test * pi;
  pi = foo (7);
  return pi->i;
}
