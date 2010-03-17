typedef struct {
  int bpp;
} foo;

extern foo * allo(int);

int bla(foo *i) {
  foo * j;
  j = allo(i->bpp);
  return 0;
}

int main(void) {
  foo * m;
  bla(m);
  return 0;
}
