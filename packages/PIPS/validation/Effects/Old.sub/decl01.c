/* Make sure that effects in declarations are taken into account */

void decl01()
{
  int i = 2;
  int j = i;
  int a[sizeof(i)];

  i = 2;
  j = i;
  j = sizeof(i);
}
