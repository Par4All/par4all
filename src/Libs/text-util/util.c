/*
 * $Id$
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "properties.h"

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

/******************************************************** LINE MANAGEMENT */

#define LINE_SUFFIX "\n"

/* returns a possible index where to cut the string. 0 if none.
 */
static int
last_comma_or_clopar(string s)
{
    int last = 0, i;
    for (i=strlen(s)-1; i>0 && !last; i--)
	if (s[i] == ',' || s[i]==')') 
	    last = i;
    return last;
}

void
add_to_current_line(
    string crt_line,     /* current line being processed */
    string add_string,   /* string to add to this line */
    string continuation, /* prefix when breaking a line */
    text txt             /* where to append completed lines */)
{
    int lcrt_line, ladd_string;

    lcrt_line = strlen(crt_line);
    ladd_string = strlen(add_string);

    if (lcrt_line + ladd_string + 2 > MAX_LINE_LENGTH) /* 2 = strlen("\n\0") */
    {   
	/* cannot append the string. must go next line.
	 */
	bool divide;
	char tmp[MAX_LINE_LENGTH];
	int last_cut = last_comma_or_clopar(crt_line);

	divide = (last_cut > 0)
	    && (last_cut != lcrt_line-1) 
	    &&  (lcrt_line - last_cut + ladd_string < MAX_LINE_LENGTH - 2);

	if (divide) 
	{
	    /* only to remain coherent with the last validation. CA.
	     */
	    if (crt_line[last_cut+1]==' ')
		last_cut++;

	    strcpy(tmp, crt_line+last_cut+1); /* save the end of the line */
	    crt_line[last_cut+1] = '\0';      /* trunc! */
	}
	
	/* happend new line. */
	strcat(crt_line, LINE_SUFFIX);
	ADD_SENTENCE_TO_TEXT
	    (txt, make_sentence(is_sentence_formatted, strdup(crt_line)));

	/* now regenerate the beginning of the line */
	strcpy(crt_line, continuation);
	if (divide) strcat(crt_line, tmp); /* get back saved part */
    }

    if (strlen(crt_line) + ladd_string + 2 > MAX_LINE_LENGTH)
	/* this shouldn't happen. 
	 */
	pips_internal_error("something got wrong...");

    strcat(crt_line, add_string);
}

void
close_current_line(
    string crt_line,
    text txt)
{
    if (strlen(crt_line)!=0)
    {
	strcat(crt_line, LINE_SUFFIX); /* should be large enough. */
	ADD_SENTENCE_TO_TEXT
	    (txt, make_sentence(is_sentence_formatted, strdup(crt_line)));
	crt_line[0] = '\0';
    }
}
