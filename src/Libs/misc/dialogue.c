/* 
  these versions of functions 'dialogue', 'message', ... are used to replaced
  XView versions when a batch version of pips is compiled.
*/

#include <stdio.h>
#include <stdarg.h>

/*VARARGS0*/
void dialogue(char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    (void) vfprintf(stderr, fmt, args);
    va_end(args);
}

/*VARARGS0*/
void show_message(char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    (void) vfprintf(stderr, fmt, args);
    va_end(args);
}
