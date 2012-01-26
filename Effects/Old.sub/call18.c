#include <stdarg.h>

void call18(int i, ...)
{
  i++;
  if(i) {
    va_list pa;
    int r;
      
    va_start(pa,fmt);

    va_end(pa);
  }
}

main()
{
  int i = 1;
  int j = 2;
  int k = 3;

  call18(i, i, j ,k, 0);
}
