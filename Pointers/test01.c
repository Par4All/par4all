/* Check the handling of a simple test */

int test01()
{
  int i, j;
  int *p;

  if(i==j)
    p = &i;
  else
    p = &j;

  return 0;
}
