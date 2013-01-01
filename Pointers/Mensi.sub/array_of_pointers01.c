/* Pointer q added to make sure that, j being unknown for the
   analysis, it is found pointing toward i and undefined. */

int main()
{
  int *a[10], *q;
  int i = 0, j = 2;
  a[0] = &i;
  q = a[j];
  a[1] = &i;
  a[2] = &i;
  a[3] = &i;
  a[4] = &i;
  a[5] = &i;
  a[6] = &i;

 return 0;
}
