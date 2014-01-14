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
/*
 * The motivation of text is to delay string concatenation by dealing with
 * lists of strings instead as more as possible.
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "arithmetique.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"


char *
i2a(int i)
{
    char *buffer;
    asprintf(&buffer,"%d", i);
    return buffer;
}


char *
f2a(float f)
{
    char *buffer;
    asprintf(&buffer, "%f", f);
    return buffer;
}

void
add_one_unformated_printf_to_text(text r,
                                  string a_format, ...)
{
  /* beurk... */
   char *buffer;
   va_list some_arguments;
   va_start(some_arguments, a_format);
   (void) vasprintf(&buffer, a_format, some_arguments);
   ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted,
                                         buffer));
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
    const char* append,       /* string to add to this line */
    string continuation, /* prefix when breaking a line */
    text txt             /* where to append completed lines */)
{
    bool divide;
    char tmp[MAX_LINE_LENGTH];
    int last_cut;
    int lappend;
    int lbuffer = strlen(buffer);
    bool comment = false;
    char stmp;
    /* special case: appends a sole "," on end of line... */
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
    comment = (stmp == 'c'|| stmp == 'C'	|| stmp == '!'|| stmp == '*'
      || (continuation[0] == '/' && continuation[1] == '/')
      || (continuation[0] == '/' && continuation[1] == '*'));
    
    if (strlen(buffer) + lappend + 2 > MAX_LINE_LENGTH) {
	/* this shouldn't happen. 
	 * it can occur if lappend+lcontinuation is too large.
	 */
	if (comment) {
	    /* Cut the comment */
	    int coupure = MAX_LINE_LENGTH-2 -lbuffer;
	    char tmp2[2*MAX_LINE_LENGTH];
	    strcpy(tmp2,append+coupure);
        /*SG: warning: I removed the const modifier here */
	    ((char*)append)[coupure]='\0';
	    
	    add_to_current_line(buffer,append,continuation,txt);
	    add_to_current_line(buffer,tmp2,continuation,txt);
	    
	}
	else 
	    pips_internal_error("line code too large...");
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
{
  if (strlen(buffer)!=0) /* do not append an empty line to text */ {
    int lbuffer=0;
    char stmp = continuation[0];
    char stmp1 = continuation[1];
    bool comment = stmp == 'c'|| stmp == 'C'
      || stmp == '!'|| stmp == '*'
      || (stmp == '/' && stmp1 == '*') || (stmp == '/' && stmp1 == '/');

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
      else
	pips_assert("buffer is too large",
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

/* Add the word list wl to the end of the last sentence of text t */
void add_words_to_text(text t,list wl)
{
  list sl = text_sentences(t);

  if(ENDP(sl)) {
    pips_internal_error("what kind of sentence to make?");
  }
  else {
    sentence s = SENTENCE(CAR(gen_last(sl)));
    if(sentence_formatted_p(s)) {
      pips_internal_error("Not implemented");
    }
    else {
      unformatted u = sentence_unformatted(s);
      unformatted_words(u) = gen_nconc(unformatted_words(u), wl);
    }
  }
  pips_assert("t is consistent", text_consistent_p(t));
}
