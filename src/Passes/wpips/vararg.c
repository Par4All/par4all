#include <stdio.h>
#include <varargs.h>
#include <setjmp.h>

#define SMALL_BUFFER_LENGTH 256

/*VARARGS0*/
void wpips_user_error(args)
va_list * args;
{
    char *fmt;
    char error_buffer[SMALL_BUFFER_LENGTH];
    char * perror_buffer = &error_buffer[0];
    extern jmp_buf pips_top_level;

    /* print name of function causing error */
    (void) sprintf(perror_buffer, "user error in %s: ", 
		    va_arg(* args, char *));
    fmt = va_arg(* args, char *);

    /* print out remainder of message */
    (void) vsprintf(perror_buffer+strlen(perror_buffer), fmt, *args);
    /* va_end(args); */

    wpips_user_error_message(error_buffer);
}


/*VARARGS0*/
void wpips_user_warning(args)
va_list * args;
{
    char *fmt;
    char warning_buffer[SMALL_BUFFER_LENGTH];

    /* print name of function causing warning */
    (void) sprintf(warning_buffer, "user warning in %s: ", 
		    va_arg(* args, char *));
    fmt = va_arg(* args, char *);

    /* print out remainder of message */
    (void) vsprintf(warning_buffer+strlen(warning_buffer), fmt, *args);
    /* va_end(args); */

    wpips_user_warning_message(warning_buffer);
}
