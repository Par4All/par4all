/* bug seen in Transformations/eval.c */

int foo(int j, int n)
{
  if(n>0)
    return 1*j;
  else if(n<0)
    return (n-n+1)*j;
  else
    return (2*n-2*n+3)*j*(n-n+2);
}
