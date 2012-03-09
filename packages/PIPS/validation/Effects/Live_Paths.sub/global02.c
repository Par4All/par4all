int N =10;

int foo (void) {
  return N;
}

int main () {
  int a = 0;
  int b = 0;
  a = foo ();
  b = foo ();
  return a + b;
}
