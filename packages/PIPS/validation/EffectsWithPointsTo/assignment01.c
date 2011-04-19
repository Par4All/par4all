/* Check the handling of multiple assignments */

void assignment01()
{
  int i;
  int * p;
  int * q;
  int * r;

  p = q = r = &i;
  i = 2;
}
