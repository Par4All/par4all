
int dowhile01()
{
  int i, x, y, a[10];
  
  i=0;
  x=10;
  y=1;
  
  i=0;
  do
  { 
    y=0;
    i++;
  }while (i<x);
  
  i=0;
  do
  { 
    a[i]=0;
    i++;
  }while (i<x);
  
  return 0;
}
