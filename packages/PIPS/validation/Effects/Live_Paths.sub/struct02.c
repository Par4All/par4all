// test liveness on struct with array as a menber

typedef struct {
  int a[5];
  float b;
} my_struct;

int main () {
  my_struct c;
  int i;

  for (i = 0; i < 5; i++) {
    c.a[i] = i;
  }
  c.b = 10.0;

  return c.a[2];
}
