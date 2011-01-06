// Check that transformers stay less precise when they are computed
// without context (i.e. block_to_transformer has no unexpected
// effect), although the preconditions end up precise for k in
// both cases. Note however that precise information about m is
// available in the second case only.

// check that refine_transformers capture the behavior of multiply:
// l is known before "return k;". Here interprocedural information is
// needed and it is not available when computing transformers in
// context, with intraprocedural information

int multiply(i,j)
{
  return i*j;
}

int twice(int j)
{
  int n = 2;
  return n*j;
}

int block01(int j)
{
  int i = 2;
  int k = i*j;
  int l = multiply(i,j);
  int m = twice(j);
  return k;
}
