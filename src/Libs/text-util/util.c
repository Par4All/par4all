/* 	%A% ($Date: 1995/12/07 15:30:55 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
static char vcid[] = "%A% ($Date: 1995/12/07 15:30:55 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "genC.h"
#include "text.h"
#include "text-util.h"

char *
i2a(int i)
{
   static char buffer[32];
   sprintf(buffer, "%d", i);
   return(strdup(buffer));
}    


char *
f2a(float f)
{
   static char buffer[32];
   sprintf(buffer, "%f", f);
   return(strdup(buffer));
}    


void
add_one_unformated_printf_to_text(text r,
                                  string a_format, ...)
{
   char buffer[200];
   
   va_list some_arguments;

   va_start(some_arguments, a_format);
   
   (void) vsprintf(buffer, a_format, some_arguments);
   ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
                                         strdup(buffer)));

   va_end(some_arguments);
}


