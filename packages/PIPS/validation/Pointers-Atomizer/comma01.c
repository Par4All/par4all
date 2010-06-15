/* Check the handling of multiple assignments */

void comma01()
{
  int i;
  int * p;
  int * q;
  int * r;

  p = &i, q = &i, r = &i;
  i = 2;
}
