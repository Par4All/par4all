// Verify that we return j=0


int main()
{
  int i0=0, i1=1, j=-1;
  int *p, *q;
  p=&i1; q=&i0;
  
  j = (0?*p:*q);
  
  return j;
}
