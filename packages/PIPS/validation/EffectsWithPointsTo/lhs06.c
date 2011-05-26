#include <stdarg.h>

int lhs06(const char * fmt,
	  void (* my_fputc)(const char c,void * _stream),
	  void * _stream,
	  va_list pa)
{
  int nbout;
  int col = 0;
  int indent = 0;
  int ui32;
  int i32;
  static char buffer[2000];

  if(! my_fputc) return(0);

  nbout = 0;

  * (__builtin_va_arg(pa,int *)) = col;
}
