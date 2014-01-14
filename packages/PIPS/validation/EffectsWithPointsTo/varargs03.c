#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

int varargs03(FILE * stream,const char * fmt,...)
{
  if(stream)
    {
      va_list pa;
      int r;
      
      va_start(pa,fmt);
      r = vfprintf(stream,fmt,pa);
      va_end(pa);
      
      return(r);
    }
  else return(0);
}

int main()
{
  FILE * s;
  char * fmt;
  char * ch;
  int tab[5];
  int r;
  r = varargs03(s, fmt, ch, tab);
  return r;
}
