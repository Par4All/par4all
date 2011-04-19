/* To be out of Emami's patterns with a user call */

int * bar(int j)
{
  return &j;
}

void assign05()
{
  int * r;
  int i;

  r = bar(i);
  i = 1;
  *r = 0;
}

void foo()
{
  assign05();
}
