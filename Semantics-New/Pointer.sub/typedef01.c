
typedef unsigned char BYTE;

int main()
{
  BYTE a[10];
  BYTE *p, *q;
  
  q=a;
  p=q+1;
  
  *q = 0;
  *p = 0;
  
  return 0;
}
