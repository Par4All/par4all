/* 	%A% ($Date: 1996/06/12 16:38:24 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_text_util[] = "%A% ($Date: 1996/06/12 16:38:24 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"

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


/* Return the first word of a sentence: */
string
first_word_of_sentence(sentence s)
{
    string begin;

    if (sentence_formatted_p(s))
	/* The sentence is seen as a big word: */
	begin = sentence_formatted(s);
    else if (sentence_unformatted_p(s)) {
	list l = unformatted_words(sentence_unformatted(s));
	/* From the first word to the last one. Label should is
           skipped: */
	begin = STRING(CAR(l));
    }
    else
	pips_assert("s should be formatted or unformatted...", 0);

    return begin;
}


/* Return the last word of a sentence: */
string
last_word_of_sentence(sentence s)
{
    string end;

    if (sentence_formatted_p(s))
	/* The sentence is seen as a big word: */
	end = sentence_formatted(s);
    else if (sentence_unformatted_p(s)) {
	list l = unformatted_words(sentence_unformatted(s));
	/* From the first word to the last one. Label should is
           skipped: */
	end = STRING(CAR(gen_last(l)));
    }
    else
	pips_assert("s should be formatted or unformatted...", 0);

    return end;
}
