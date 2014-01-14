// Verify that we return *p=i=0


int main()
{
  int i = 0;
  int *p;
  
  p=&i;
  
  return *p;
}
