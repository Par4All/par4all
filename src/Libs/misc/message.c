/*
 * $Id$
 * 
 * error & message handling handling routines (s.a. debug.c) :
 * user_log()
 * user_request()
 * user_warning()
 * user_error()
 * pips_error()
 * pips_assert()
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

#include "genC.h"
#include "misc.h"

/* pips meta information from pipsmake are made available here...
 * (current phase and module...)
 */
static string current_phase = NULL;
static string current_module = NULL;

void set_pips_current_computation(string rname, string oname)
{
    pips_assert("no current computation", !current_module && !current_phase);

    current_phase = strdup(rname);
    current_module = strdup(oname);
}

void reset_pips_current_computation(void)
{
    pips_assert("some current computation", current_module && current_phase);

    free(current_module), current_module = NULL;
    free(current_phase), current_phase = NULL;
}

/* FI: grah, qu'est-ce que c'est que cette heresie? misc ne devait pas
   dependre de NewGen! */
extern bool get_bool_property(string);

#define INPUT_BUFFER_LENGTH 256 /*error caught by terminal at 257th character*/


/* USER_LOG is a function that should be called to log the current
 * PIPS request, as soon as it is relevant. loging will occure if property
 * USER_LOG_P is TRUE. USER_LOG should be called as:
 *
 * USER_LOG(format [, arg] ... )
 *
 * where format and arg-list are passed as arguments to vprintf.  
 */

void default_user_log(char * fmt, va_list args)
{
    if(!get_bool_property("USER_LOG_P")) return;
    (void) vfprintf(stdout, fmt, args);
    fflush(stdout);
}

/* default assignment of pips_log_handler is default_user_log. Some 
 * top-level (eg. wpips) may need a special user_log proceedure; they 
 * should let pips_log_handler point toward it.
 *
 * Every procedure pointed to by pips_log_handler must test the property 
 * USER_LOG_P; this is necessary because (* pips_log_handler) may be called 
 * anywhere (because VARARGS), whithout verifying it.
 */
void (* pips_log_handler)(char * fmt, va_list args) = default_user_log;


/* USER_LOG(format [, arg] ... ) */
/*VARARGS1*/
void user_log(char * a_message_format, ...)
{
    va_list some_arguments;
    va_start(some_arguments, a_message_format);
    (* pips_log_handler)(a_message_format, some_arguments);
    va_end(some_arguments);
}

/* USER_REQUEST is a function that should be called to request some data 
 * from the user. It returns the string typed by the user until the 
 * return key is typed.
 * USER_REQUEST should be called as:
 *
 * USER_REQUEST(format [, arg] ... )
 *
 * where format and arg-list are passed as arguments to vprintf.  
 */

string default_user_request(fmt, args)
char *fmt;
va_list args;
{
    printf("\nWaiting for your response: ");
    vfprintf(stdout, fmt, args);
    fflush(stdout);
    return safe_readline(stdin);
}

/* default assignment of pips_request_handler is default_user_request. Some 
 * top-level (eg. wpips) may need a special user_request proceedure; they 
 * should let pips_request_handler point toward it.
 */
string (* pips_request_handler)(char *, va_list) = default_user_request;

/* The generic fonction to ask something to the user. Note that if
 * the user cancels his/her request,the empty string "" is returned:
 */
string user_request(char * a_message_format, ...)
{
   string str;
   va_list some_arguments;
   va_start(some_arguments, a_message_format);
   str = (* pips_request_handler)(a_message_format, some_arguments);
   va_end(some_arguments);
   return(str);
}


/* USER_WARNING issues a warning and don't stop the program (cf. user_error
 * for infos.) 
 */

static FILE * warning_file = (FILE*) NULL;
static string warning_file_name = (string) NULL;

#define WARNING_FILE_NAME "Warnings"

void open_warning_file(string dir)
{
    warning_file_name = strdup(concatenate(dir, "/", WARNING_FILE_NAME, 0));
    warning_file = safe_fopen(warning_file_name, "a");
}

void close_warning_file(void)
{
    if (warning_file) 
    {
	safe_fclose(warning_file, warning_file_name);

	warning_file = (FILE*) NULL;
	free(warning_file_name);
	warning_file_name = (string) NULL;
    }
}

static void 
append_to_warning_file(char * calling_function_name,
		       char * a_message_format,
		       va_list * some_arguments)
{
   if (warning_file) 
   {
     fprintf(warning_file, "%s[%s] (%s) ", 
	     current_phase? current_phase: "unknown",
	     current_module? current_module: "unknown",
	     calling_function_name);
     vfprintf(warning_file, a_message_format, *some_arguments);
   }
}

static void
default_user_warning(char * calling_function_name,
                     char * a_message_format,
                     va_list * some_arguments)
{
   /* print name of function causing warning
    * print out remainder of message 
    */
    fprintf(stderr, "user warning in %s: ", calling_function_name);
    vfprintf(stderr, a_message_format, *some_arguments);
}

/* default assignment of pips_warning_handler is default_user_warning. Some 
 * top-level (eg. wpips) may need a special user_warning proceedure; they 
 * should let pips_warning_handler point toward it.
 */
void (* pips_warning_handler)(char *, char *, va_list *) 
     = default_user_warning;


void
user_warning(char * calling_function_name,
             char * a_message_format,
             ...)
{
   va_list some_arguments;

   if (get_bool_property("NO_USER_WARNING")) return; /* FC */

   /* WARNING FILE */
   va_start(some_arguments, a_message_format);
   append_to_warning_file
       (calling_function_name, a_message_format, &some_arguments);
   va_end(some_arguments);

   /* STDERR or whatever... */
   va_start(some_arguments, a_message_format);
   (* pips_warning_handler)
       (calling_function_name, a_message_format, &some_arguments);
   va_end(some_arguments);
}

/* if not GNU C */
#if !defined(__GNUC__)
void 
pips_user_warning_function(
    char * format,
    ...)
{
    va_list args;
    if (get_bool_property("NO_USER_WARNING")) return; /* FC */
    va_start(args, format);
    (* pips_warning_handler)(pips_unknown_function, format, &args);
    va_end(args);
}

void 
pips_user_error_function(
   char * format,
   ...)
{
    va_list args;
    va_start(args, format);
    (*pips_error_handler)(pips_unknown_function, format, &args);
    va_end(args);
}

void 
pips_internal_error_function(
   char * format,
   ...)
{
   va_list some_arguments;
   (void) fprintf(stderr, "pips error in %s: ", pips_unknown_function);
   va_start(some_arguments, format);
   (void) vfprintf(stderr, format, some_arguments); /* ??? */
   va_end(some_arguments);

   /* create a core file for debug */
   (void) abort();
}

#endif /* no __GNUC__ */

/* make sure the user has noticed something */
void default_prompt_user(string s)
{
    fprintf(stderr, "%s\nPress <Return> to continue ", s);
    while (getchar() != '\n') ;
}

void pips_exit_function(int code, char * format, ...)
{
    va_list some_arguments;
    va_start(some_arguments, format);
    (void) vfprintf(stderr, format, some_arguments); /* ??? */
    va_end(some_arguments);
    exit(code);
}

/* PROMPT_USER schould be implemented. (its a warning with consent of the user)
The question is: how schould it be called?
*/

/* USER_ERROR is a function that should be called to terminate the current
PIPS request execution or to restore the previous saved stack environment 
when an error in a Fortran file or elsewhere is detected.
USER_ERROR first prints on stderr a description of the error, then tests 
the property ABORT_ON_USER_ERROR and aborts case true. Else it will restore 
the last saved stack environment (ie. come back to the last executed 
setjmp(pips_top_level) ), except for eventuality it has already been 
called. In this case, it terminates execution with exit.
USER_ERROR should be called as:

   USER_ERROR(fonction, format [, arg] ... )

where function is a string containing the name of the function
calling USER_ERROR, and where format and arg-list are passed as
arguments to vprintf.

Modifications:
 - user_error() was initially called when a Fortran syntax error was
   encountered by the parser; execution was stopped; this had to be changed
   because different kind of errors can occur and because pips is no longer
   structured using exec(); one has to go back to PIPS top level, in tpips
   or in wpips; (Francois Irigoin, 19 November 1990)
 - user_error() calls (* pips_error_handler) which can either be 
   default_user_error or a user_error function specific to the used top-level.
   But each user_error function should have the same functionalities.
*/

static void
default_user_error(char * calling_function_name,
                   char * a_message_format,
                   va_list *some_arguments)
{
   /* print name of function causing error */
   (void) fprintf(stderr, "user error in %s: ", calling_function_name);

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, * some_arguments);

   /* terminate PIPS request */
   /* here is an issue: what if the user error was raised from properties */
   if (get_bool_property("ABORT_ON_USER_ERROR")) 
       abort();
   else {
      static int user_error_called = 0;

      if (user_error_called > 2) {
         (void) fprintf(stderr, "This user_error is too much! Exiting.\n");
         exit(1);
      }
      else {
         user_error_called++;
      }

      /* throw according to linear exception stack! */
      THROW(user_exception_error);
   }
}

/* default assignment of pips_error_handler is default_user_error. Some 
 * top-level (eg. wpips) may need a special user_error proceedure; they 
 * should let pips_error_handler point toward it.
 */
void (* pips_error_handler)(char *, char *, va_list *) = default_user_error;

void
user_error(char * calling_function_name,
           char * a_message_format,
           ...)
{
   va_list some_arguments;
   va_start(some_arguments, a_message_format);
   (* pips_error_handler)
       (calling_function_name, a_message_format, &some_arguments);
   va_end(some_arguments);
}


/* PIPS_ERROR is a function that should be called to terminate PIPS
execution when data structures are corrupted. PIPS_ERROR should be
called as:
  PIPS_ERROR(fonction, format [, arg] ... )
where function is a string containing the name of the function
calling PIPS_ERROR, and where format and arg-list are passed as
arguments to vprintf. PIPS_ERROR terminates execution with abort.
*/

void
pips_error(
    char * calling_function_name,
    char * a_message_format,
    ...)
{
   va_list some_arguments;
   (void) fprintf(stderr, "pips error in %s: ", calling_function_name);
   va_start(some_arguments, a_message_format);
   (void) vfprintf(stderr, a_message_format, some_arguments);
   va_end(some_arguments);

   /* create a core file for debug */
   (void) abort();
}

/* PIPS_ASSERT tests whether the second argument is true. If not, message
is issued and the program aborted. The first argument is the function name.
  pips_assert(function_name, boolean);
*/

void 
pips_assert_function(
    char * function, /* the name of the function if available */
    int predicate,   /* predicate to be tested */
    int line,        /* location of the assertion */
    char * file)     /* location of the assertion */
{
    /* print name of function causing error and
     * create a core file for debug 
     */
    if(!predicate) 
	(void) fprintf(stderr, "pips assertion failed"
		       " in function %s at line %d of file %s\n",
		       function, line, file),
	(void) abort();
}

void
user_irrecoverable_error(char * calling_function_name,
			 char * a_message_format,
			 ...)
{
   va_list some_arguments;
   va_start(some_arguments, a_message_format);
   /* print name of function causing error */
   (void) fprintf(stderr, "user error in %s: ", calling_function_name);

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, some_arguments);

   exit(1);
   va_end(some_arguments);
}

void
user_irrecoverable_error_function(char * a_message_format, ...)
{
   va_list some_arguments;
   va_start(some_arguments, a_message_format);
   /* print name of function causing error */
   (void) fprintf(stderr, "user error in UNKNOW...: ");

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, some_arguments);

   exit(1);
   va_end(some_arguments);
}
