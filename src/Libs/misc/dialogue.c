/* 
  these versions of functions 'dialogue', 'message', ... are used to replaced
  XView versions when a batch version of pips is compiled.
*/

#include <stdio.h>
#include <varargs.h>
#include <setjmp.h>

/*VARARGS0*/
void dialogue(va_alist)
va_dcl
{
    va_list args;
    char *fmt;

    va_start(args);

    fmt = va_arg(args, char *);

    (void) vfprintf(stderr, fmt, args);

    va_end(args);
}

/*VARARGS0*/
void show_message(va_alist)
va_dcl
{
    va_list args;
    char *fmt;

    va_start(args);

    fmt = va_arg(args, char *);

    (void) vfprintf(stderr, fmt, args);

    va_end(args);
}
