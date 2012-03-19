// test liveness on struct

typedef struct {
  int a;
  float b;
} my_struct;

int main () {
  my_struct c;

  c.a = 10;
  c.b = 10.0;

  return c.a;
}
