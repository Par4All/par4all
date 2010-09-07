typedef struct {
  int in1;
  int in2;
  int out1;
} ex;

ex var;
void add () {
  var.out1 = var.in1 + var.in2;
}

int main ()
{
  var.in1 = 0;
  var.in2 = 1;
  return var.out1;
}
