// On souhaite modifier i de différente facon


int main()
{
  int i = 0;
  int *p;
  int j = 10;
  int *q;
  
  p = &i;
  
  q = &j;
  
  //on modifie i
  i=*q;
  
  return 0;
}
