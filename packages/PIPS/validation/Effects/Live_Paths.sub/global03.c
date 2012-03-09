int N =10;

int foo (int v) {
  N = v;
  return N;
}

int main () {
  int a = 0;
  int b = 0;
  a = foo (a);
  b = foo (b);
  return a + b;
}
