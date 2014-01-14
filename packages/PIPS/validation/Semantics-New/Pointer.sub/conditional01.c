// Verify that we return j=0


int main()
{
  int i = 0, j=-1;
  int *p;
  p=&i;
  
  j = ((*p=0)?0:1);
  
  return j;
}
