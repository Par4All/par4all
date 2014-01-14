
typedef unsigned char BYTE;
typedef unsigned char *P_BYTE;


int main()
{
  BYTE a[10], b[10];
  P_BYTE p0, q1;
  BYTE *q0, *p1;
  
  p0=a;
  q0=p0+1;

  *p0 = 0;
  *q0 = 0;
  
  
  p1=b;
  q1=p1+1;
  
  *p1 = 0;
  *q1 = 0;
  
  return 0;
}
