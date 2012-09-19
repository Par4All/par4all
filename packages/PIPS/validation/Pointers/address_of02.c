/* Same as address_of01.c, but the mistake is hidden */

int foo()
{
  int *p, i[10];
  p = (int *) i;
  p++;
  p++;
  return 0;
}


 

