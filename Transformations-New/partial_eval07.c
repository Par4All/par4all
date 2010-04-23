/* bug seen in Expressions/partial_eval01.c, but not reproduced here */

int foo(int i)
{
  int j;
  int n;

  if(n>0)
    return j = i++ + 0;
  else if(n<0)
    return i++ + 0;
  else
    return 1+2+3;
}
