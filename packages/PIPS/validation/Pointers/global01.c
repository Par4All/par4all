/* Synthesize stubs for global pointers for intraprocedural analysis */

int * i;

int * j;

int k;

int global01()
{
  i = &k;
  j = i;
  return 0;
}
