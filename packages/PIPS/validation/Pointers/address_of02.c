/* Same as address_of01.c, but the mistake is hidden */

int foo()
{
  int *p, i=1;
  p = &i;
  p++;
  p++;
  return 0;
}


 

