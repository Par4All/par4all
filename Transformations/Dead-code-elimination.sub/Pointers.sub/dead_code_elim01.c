/* test case for use-def elimination suggested by François for Mehdi.*/
int dead_code_elim01()
{
  int i, *x, *y;

  i = 2;
  x = &i;
  y = x;
  *y = 1;
  
  return *y;
}

