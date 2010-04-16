/* Impact of dynamic aliasing */

int pointer03()
{
  int i = 3;
  int * ip = &i;
  *ip = 4;
  return i;
}
