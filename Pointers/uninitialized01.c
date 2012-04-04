/* Dereferencement of an uninitialized pointer */

void uninitialized01()
{
  int ** pp;
  int i;
  *pp = &i;
  return;
}
