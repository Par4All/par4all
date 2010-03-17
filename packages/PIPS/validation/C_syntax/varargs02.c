#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

int varargs02(FILE * stream,const char * fmt,...)
{
  if(stream)
    {
      va_list pa;
      int r;
      
      va_start(pa,fmt);
      r = va_arg(pa, int);
      va_end(pa);
      
      return(r);
    }
  else return(0);
}
