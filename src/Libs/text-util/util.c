/*
 * $Id$
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "arithmetique.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "properties.h"

#define CONTINUATION PIPS_COMMENT_CONTINUATION "                              "
char *
i2a(int i)
{
    char buffer[32];
    sprintf(buffer, "%d", i);
    return strdup(buffer);
}    


char *
f2a(float f)
{
    char buffer[32];
    sprintf(buffer, "%f", f);
    return strdup(buffer);
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
    string buffer,       /* current line being processed */
    string append,       /* string to add to this line */
    string continuation, /* prefix when breaking a line */
    text txt             /* where to append completed lines */)
{
    bool divide;
    char tmp[MAX_LINE_LENGTH];
    int last_cut;
    int lappend;
    int lbuffer = strlen(buffer);
    boolean comment = FALSE;
    char stmp;
    /* spacial case: appends a sole "," on end of line... */
    if (same_string_p(append, ", ") && lbuffer+3==MAX_LINE_LENGTH) 
	append = ",";

    lappend = strlen(append);


    if (lbuffer + lappend + 2 > MAX_LINE_LENGTH) /* 2 = strlen("\n\0") */
    {   
	/* cannot append the string. must go next line.
	 */
	last_cut = last_comma_or_clopar(buffer);

	divide = (last_cut > 0)
	    && (last_cut != lbuffer-1) 
	    && (lbuffer - last_cut + lappend +strlen(continuation) 
		< MAX_LINE_LENGTH - 2);

	if (divide) 
	{
	    int nl_cut = last_cut;
	    while (buffer[nl_cut+1]==' ') nl_cut++;
	    strcpy(tmp, buffer+nl_cut+1); /* save the end of the line */
	    buffer[last_cut+1] = '\0';    /* trunc! */
	}
	
	/* happend new line. */
	strcat(buffer, LINE_SUFFIX);
	ADD_SENTENCE_TO_TEXT
	    (txt, make_sentence(is_sentence_formatted, strdup(buffer)));
	
	/* now regenerate the beginning of the line */
	strcpy(buffer, continuation);
	
	if (divide) {
	    strcat(buffer, tmp); /* get back saved part */
	    
	    if (strlen(buffer) + lappend + 2 > MAX_LINE_LENGTH
		&& ! same_string_p(buffer,continuation)) {
		/* current line + new line are too large. Try to append the 
		   buffer alone, before trying to add the new line alone */
		strcat(buffer, LINE_SUFFIX);
		ADD_SENTENCE_TO_TEXT
		    (txt, make_sentence(is_sentence_formatted, strdup(buffer)));
		strcpy(buffer, continuation);
	    }
	}
	
    }
    
    /* Append the new line */
    lbuffer = strlen(buffer);
    stmp = continuation[0];
    comment = stmp == 'c'|| stmp == 'C'	|| stmp == '!'|| stmp == '*';
    
    if (strlen(buffer) + lappend + 2 > MAX_LINE_LENGTH) {
	/* this shouldn't happen. 
	 * it can occur if lappend+lcontinuation is too large.
	 */
	if (comment) {
	    /* Cut the comment */
	    int coupure = MAX_LINE_LENGTH-2 -lbuffer;
	    char tmp2[2*MAX_LINE_LENGTH];
	    strcpy(tmp2,append+coupure);
	    append[coupure]='\0';
	    
	    add_to_current_line(buffer,append,continuation,txt);
	    add_to_current_line(buffer,tmp2,continuation,txt);
	    
	}
	else 
	    pips_internal_error("line code too large...\n");
    }
    else if (! same_string_p(append, " ") 
	     || ! same_string_p(buffer, continuation))
	strcat(buffer, append);
}

void
close_current_line(
    string buffer,
    text txt,
    string continuation)
{  if (strlen(buffer)!=0) /* do not append an empty line to text */
    {
	int lbuffer=0; 
	char stmp = continuation[0];
	boolean comment = stmp == 'c'|| stmp == 'C'	
	|| stmp == '!'|| stmp == '*';


	if ((lbuffer=strlen(buffer))+2>MAX_LINE_LENGTH) { 
	    if (comment) {
		int coupure = MAX_LINE_LENGTH-2;
		char tmp2[2*MAX_LINE_LENGTH];
		strcpy(tmp2,buffer+coupure);
		buffer[coupure]='\0';
		
		add_to_current_line(buffer,"  ",continuation,txt);
		add_to_current_line(buffer,tmp2,continuation,txt);
		close_current_line(buffer,txt,continuation);
	    }
	    else pips_assert("buffer is too large", 
			     strlen(buffer)+1<MAX_LINE_LENGTH);
	}
	else {
	    strcat(buffer, LINE_SUFFIX);
	    ADD_SENTENCE_TO_TEXT
		(txt, make_sentence(is_sentence_formatted, strdup(buffer)));
	    buffer[0] = '\0';
	    
	}
    }

}


