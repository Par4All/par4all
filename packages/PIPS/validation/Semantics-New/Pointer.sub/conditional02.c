// Verify that we return j=0


int main()
{
  int i0=0, i1=1, j=-1;
  int *p, *q;
  p=&i0; q=&i1;
  
  j = (1?*p:*q);
  
  return j;
}
