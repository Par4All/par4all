/* To make sure that the transformer for the comma expression is OK  */

int comma07()
{
  int i, j;

  i = (j = 1, 2);
  i = 3, j = 4;
  return i;
}

