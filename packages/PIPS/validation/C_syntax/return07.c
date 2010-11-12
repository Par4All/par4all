/* Check that gotos and assignments are generated to represent C
   return statements when the value returned is not as simple as an int */

typedef struct {int a;} a_t;

a_t return07()
{
  int i, j;
  a_t a1;
  if(i) {
    j = 1;
    return a1;
  }
  else {
    j = 2;
    return a1;
  }
  j = 3;
  return a1;
}
