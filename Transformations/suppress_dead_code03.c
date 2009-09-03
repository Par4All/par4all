/* check that user-defined calls are not eliminated as dead code
   because they do not write the store. Yes, they may have some control
   effects! */

void foo(int i)
{
  ;
}

int suppress_dead_code03(int i)
{
  int k;

  foo(1);
  k = 3;

  return i;
} 

int main()
{
  return suppress_dead_code03(2);
}
