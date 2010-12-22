
struct test {
  int i;
  int j;
};

struct test foo () {
  struct test result;
  result.i = 0;
  result.j = 0;
  return result;
}


int main () {
  struct test i;
  i = foo ();
  return i.i;
}
