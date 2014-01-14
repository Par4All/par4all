
typedef unsigned char BYTE;
typedef unsigned char *P_BYTE;


int main()
{
  BYTE a[10];
  P_BYTE p, q;
  
  q=a;
  p=q+1;
  
  *q = 0;
  *p = 0;
  
  return 0;
}
