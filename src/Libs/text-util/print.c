#include <stdio.h>
extern int fprintf();
extern int _flsbuf();
extern char * toupper();
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
	else {
	    fputs("      ", fd);
	}

	for (i = 0; i < em; i++) putc(' ', fd);
	col = 7+em;

	while (lw) {
	    string w = STRING(CAR(lw));

	    STRING(CAR(lw)) = NULL;
	    lw = CDR(lw);

	    /* if the string fits on the current line: no problem */
	    if (col + strlen(w) <= 70) {
		(void) fprintf(fd, "%s", w);
		col += strlen(w);
	    }
	    /* if the string fits on one line: use the 88 algorithm to break as few
	     * syntactic constructs as possible */
	    else if(strlen(w) < 70-7-em) {
		if (col + strlen(w) > 70) {
		    /* complete current line */
		    if (n > 0) {
			for (i = col; i <= 72; i++) putc(' ', fd);
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
	    /* if the string has to be broken in at least two lines: new algorithmic part
	     * to avoid line overflow (FI, March 1993) */
	    else {
		char * line = w;
		int ncar;

		/* complete the current line */
		ncar = 72 - col + 1;
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
		    ncar = MIN(72 - 7 +1, strlen(line));

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

	pips_assert("print_sentence", col <= 72);

	if (n > 0) {
	    for (i = col; i <= 72; i++) putc(' ', fd);
	    fprintf(fd, "%04d", n);
	}
	putc('\n', fd);
    }
}

void print_text(fd, t)
FILE *fd;
text t;
{
    MAPL(cs, 
	 print_sentence(fd, SENTENCE(CAR(cs))), 
	 text_sentences(t));
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


void print_words(fd, lw)
FILE *fd;
cons *lw;
{
    string s = words_to_string(lw);
    fputs(s, fd);
    free(s);
}
