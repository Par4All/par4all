/* bug seen in Expressions/partial_eval01.c */

int foo(int i)
{
  int j;
  int n;

  j = i++ + 0;

  return j;
}
