/* Synthesize stubs for global pointers for intraprocedural analysis */

int i;
// int j;
// int * p_amira=&i;
// int * p=&i;
int * p;

int k;

int global06()
{
  int * q;

  //q = p_amira;
  q = p;

  return 0;
}
