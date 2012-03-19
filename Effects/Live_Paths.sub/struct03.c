// test liveness on struct with function call and struct passed by value

typedef struct {
  int a[5];
  float b;
} my_struct;

void foo (my_struct p) {
  int i;

  for (i = 0; i < 5; i++) {
    p.a[i] = i;
  }
}

int main () {
  my_struct c;

  foo (c);

  c.b = 10.0;

  return c.a[2];
}
