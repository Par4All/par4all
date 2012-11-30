/* FI looking for recursive calls: second third of array10, bar */

double a[100];

int bar(int *p)
{
  int b[100];
  p = &b[0];
  a[*p++]= 2.;
  return 0;
}
