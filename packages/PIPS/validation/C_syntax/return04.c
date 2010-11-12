/* Check that gotos and assignments are generated to represent C
   return statements */

int return04()
{
  int i, j;
  if(i) {
    j = 1;
    return 1;
  }
  else {
    j = 2;
    return 2;
  }
  j = 3;
  return 3;
}
