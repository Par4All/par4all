/* bug seen in Transformations/eval.c */

int foo(int j, int n)
{
  return (2*n+1)*0;
}
