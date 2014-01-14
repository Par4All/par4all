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
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "genC.h"
/* #include "ri.h" */
#include "text.h"

#include "misc.h"
/* #include "ri-util.h" */

#include "arithmetique.h"


/* #include "constants.h" */

/* FI: just to make sure that text.h is built; pips-makemake -l does not
   tale into account a library whose modules do not use the library header */
#include "text-util.h"

/* print_sentence:
 *
 * FI: I had to change this module to handle string longer than the space available
 * on one line; I tried to preserve as much as I could of the previous behavior to
 * avoid pseudo-hyphenation at the wrong place and to avoid extensicve problems
 * with validate; the resulting code is lousy, of course; FI, 15 March 1993
 * 
 * RK: the print_sentence could print lower case letter according to
 * a property... 17/12/1993.
 */
void print_sentence(fd, s)
FILE *fd;
sentence s;
{
    if (sentence_formatted_p(s)) {
	string ps = sentence_formatted(s);
	while (*ps) {
	    char c = *ps++;
	    /* FI/FC: Why on earth?!?
	       (void) putc((islower((int) c) ? (char) toupper((int) c) : c), fd);
	       */
	    (void) putc( c, fd);
	}
    }
    else {
	unformatted u = sentence_unformatted(s);
	int col;
	int i;
	int line_num = 1;
	string label = unformatted_label(u);
	int em = unformatted_extra_margin(u);
	int n = unformatted_number(u);
	cons *lw = unformatted_words(u);

	if (label != (char *) NULL) {
	    fprintf(fd, "%-5s ", label);
	}
	else if(prettyprint_language_is_fortran_p())
	    fputs("      ", fd);
	}


	for (i = 0; i < em; i++) 
	    putc(' ', fd);
	col = 7+em;

	pips_assert("print_sentence", col <= MAX_LINE_LENGTH);

	while (lw) {
	    string w = STRING(CAR(lw));

	    STRING(CAR(lw)) = NULL;
	    lw = CDR(lw);

	    /* if the string fits on the current line: no problem */
	    if (col + strlen(w) <= 70) {
		(void) fprintf(fd, "%s", w);
		col += strlen(w);
	    }
	    /* if the string fits on one line: 
	     * use the 88 algorithm to break as few
	     * syntactic constructs as possible */
	    else if(strlen(w) < 70-7-em) {
		if (col + strlen(w) > 70) {
		    /* complete current line */
		    if (n > 0) {
			for (i = col; i <= MAX_LINE_LENGTH; i++) putc(' ', fd);
			fprintf(fd, "%04d", n);
		    }

		    /* start a new line with its prefix */
		    putc('\n', fd);

		    if(label != (char *) NULL 
		       && (strcmp(label,"CDIR$")==0
			   || strcmp(label,"CDIR@")==0
			   || strcmp(label,"CMIC$")==0)) {
			/* Special label for Cray directives */
			fputs(label, fd);
			fprintf(fd, "%d", (++line_num)%10);
		    }
		    else
			fputs("     &", fd);

		    for (i = 0; i < em; i++)
			putc(' ', fd);

		    col = 7+em;
		}
		(void) fprintf(fd, "%s", w);
		col += strlen(w);
	    }
	    /* if the string has to be broken in at least two lines: 
	     * new algorithmic part
	     * to avoid line overflow (FI, March 1993) */
	    else {
		char * line = w;
		int ncar;

		/* complete the current line */
		ncar = MAX_LINE_LENGTH - col + 1;
		fprintf(fd,"%.*s", ncar, line);
		line += ncar;
		col = 73;

		/*
		if (n > 0) {
		    for (i = col; i <= 72; i++) putc(' ', fd);
		    fprintf(fd, "%04d", n);
		}
		*/

		while(strlen(line)!=0) {
		    ncar = MIN(MAX_LINE_LENGTH - 7 +1, strlen(line));

		    /* start a new line with its prefix but no indentation
		     * since string constants may be broken onto two lines */
		    putc('\n', fd);

		    if(label != (char *) NULL 
		       && (strcmp(label,"CDIR$")==0
			   || strcmp(label,"CDIR@")==0
			   || strcmp(label,"CMIC$")==0)) {
			/* Special label for Cray directives */
			fputs(label, fd);
			(void) fprintf(fd, "%d", (++line_num)%10);
		    }
		    else
			fputs("     &", fd);

		    col = 7 ;
		    (void) fprintf(fd,"%.*s", ncar, line);
		    line += ncar;
		    col += ncar;
		}
	    }
	    free(w);
	}

	pips_assert("print_sentence", col <= MAX_LINE_LENGTH);

	if (n > 0) {
	    for (i = col; i <= MAX_LINE_LENGTH; i++) putc(' ', fd);
	    fprintf(fd, "%04d", n);
	}
	putc('\n', fd);
    }
}

void dump_sentence(sentence s)
{
    print_sentence(stderr, s);
}

void print_text(fd, t)
FILE *fd;
text t;
{
    MAPL(cs,
	 print_sentence(fd, SENTENCE(CAR(cs))),
	 text_sentences(t));
}

/* FI: print_text() should be fprint_text() and dump_text(), print_text() */

void dump_text(t)
text t;
{
    print_text(stderr, t);
}

string words_to_string(lw)
cons *lw;
{
    static char buffer[1024];

    buffer[0] = '\0';
    MAPL(pw, {
	string w = STRING(CAR(pw));
	if (strlen(buffer)+strlen(w) > 1023) {
	    fprintf(stderr, "[words_to_string] buffer too small\n");
	    exit(1);
	}
	(void) strcat(buffer, w);
    }, lw);

    return(strdup(buffer));
}

/* SG: moved here from icfdg */
string sentence_to_string(sentence sen)
{
    if (!sentence_formatted_p(sen))
        return words_to_string(unformatted_words(sentence_unformatted(sen)));
    else
        return sentence_formatted(sen);
}

/* SG: moved here from ricedg */
string text_to_string(text t)
{
  string str = strdup("");
  string str_new;
  MAP(SENTENCE, sen, {
    str_new = strdup(concatenate(str, sentence_to_string(sen), NULL));
    free(str);
    str = str_new;
  }, text_sentences(t));
  return(str);
}

void dump_words(list lw)
{
    print_words(stderr, lw);
}

/* print a list of strings */
void dump_strings(list sl)
{
  dump_words(sl);
}


void print_words(fd, lw)
FILE *fd;
cons *lw;
{
    string s = words_to_string(lw);
    fputs(s, fd);
    free(s);
}
