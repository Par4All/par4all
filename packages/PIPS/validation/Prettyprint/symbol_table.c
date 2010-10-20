int in1,in2,in3,out1;

static void add_comp_output()
{
  int dst;
  int i = 0;
  out1 = in1+in2>in3;
  dst = out1;
  for (i = 0; i < 100; i++) {
    int k = 5;
    k++;
  }
}

void main () {
  add_comp_output ();
}
