/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define SMALL_BUFFER_LENGTH 2560

/* two nice examples of static buffer overflow... */

/*VARARGS0*/
void
wpips_user_error(char * calling_function_name,
                 char * a_message_format,
                 va_list * some_arguments)
{
   char error_buffer[SMALL_BUFFER_LENGTH];

   /* print name of function causing error */
   (void) sprintf(error_buffer, "user error in %s: ", 
                  calling_function_name);

   /* print out remainder of message */
   (void) vsprintf(error_buffer + strlen(error_buffer),
                   a_message_format, *some_arguments);

   wpips_user_error_message(error_buffer);
}


/*VARARGS0*/
void
wpips_user_warning(char * calling_function_name,
                   char * a_message_format,
                   va_list * some_arguments)
{
   char warning_buffer[SMALL_BUFFER_LENGTH];

   /* print name of function causing warning */
   (void) sprintf(warning_buffer, "user warning in %s: ", 
                  calling_function_name);

   /* print out remainder of message */
   (void) vsprintf(warning_buffer+strlen(warning_buffer),
                   a_message_format, *some_arguments);

   wpips_user_warning_message(warning_buffer);
}
