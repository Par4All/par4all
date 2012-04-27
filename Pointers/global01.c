/* Synthesize stubs for global pointers for intraprocedural analysis */

int * i;

int * j;

int k;

int global01()
{
  // To avoid a problem with the semantics of the empty points-to set
  // The solution might be to add always an arc ANYWHERE->ANYWHERE
  // when entering a module statement
  int * p = &k;
  i = &k;
  j = i;
  return 0;
}
