#include <stdlib.h>

int main()
{
  int a[10], *p;
  double b[10], *q;
  p=&a[5];
  q=&b[4];
  
  if (rand()) {
    p++;
    q--;
  } else {
    p--;
    q++;
  }
  
  return 0;
}
