/* pi is written and ends up pointing to nowhere */

int formal_parameter01(int **pi)
{
  /* FI: I need the summary for the sequence, hence if(1) */
  if(1) {
    int ** q;
    int *i;
    int j;

    i = 0;
    q = pi;
    q++; // Incompatible with call site since pi points toward a scalar
    pi = &i;
    *pi = &j;
    *q = &j; // Incompatible with call site, see previous comment
  }
  return 0;
}

int main()
{
  int i, *ip, **aipp;
  ip = &i;
  aipp = &ip;
  i = formal_parameter01(aipp);
  return 0;
}
