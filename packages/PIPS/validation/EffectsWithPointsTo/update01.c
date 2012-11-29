/* Check the handling of an update */

void update01()
{
  int i[2];
  int * p;

  p = &i[0];
  p++;
  i[0] = 2;
}
