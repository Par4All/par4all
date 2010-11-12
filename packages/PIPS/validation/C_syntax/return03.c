/* Check that gotos are generated to represent C return statements */

void return03()
{
  int i, j;
  if(i) {
    j = 1;
    return;
  }
  else {
    j = 2;
    return;
  }
  j = 3;
  return;
}
