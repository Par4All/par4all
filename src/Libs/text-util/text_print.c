#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "text.h"

#include "misc.h"
#include "ri-util.h"
#include "properties.h"

/* #include "constants.h" */

/* FI: just to make sure that text.h is built; pips-makemake -l does not
   tale into account a library whose modules do not use the library header */
#include "text-util.h"


static int position_in_the_output = 0;

/* Before using print_sentence: */
static void
print_sentence_init()
{
    position_in_the_output = 0;
}


/* Output functions that tracks the number of output characters: */
static char
putc_sentence(char c,
	      FILE * fd)
{
    position_in_the_output ++;
    return putc(c, fd);
}

static int
fprintf_sentence(FILE * fd,
		 char * a_format,
		 ...)
{
    va_list some_arguments;
    int number_of_printed_char;
    
    va_start(some_arguments, a_format);
    number_of_printed_char = vfprintf(fd, a_format, some_arguments);
    va_end(some_arguments);
    
    position_in_the_output += number_of_printed_char;    
    return number_of_printed_char;
}


/* print_sentence:
 *
 * FI: I had to change this module to handle string longer than the space 
 * available on one line; 
 * I tried to preserve as much as I could of the previous behavior to
 * avoid pseudo-hyphenation at the wrong place and to avoid extensicve problems
 * with validate; the resulting code is lousy, of course; FI, 15 March 1993
 * 
 * RK: the print_sentence could print lower case letter according to
 * a property... 17/12/1993.
 */

#define MAX_END_COLUMN 		(72)
#define MAX_START_COLUMN 	(42)

void
print_sentence(FILE * fd,
	       sentence s)
{
    if (sentence_formatted_p(s)) {
	string ps = sentence_formatted(s);
	
	while (*ps) {
	    char c = *ps;
	    putc_sentence(c, fd);
	    deal_with_attachments_at_this_character(ps,
						    position_in_the_output);
	    ps++;
	}
    }
    else 
    {
	/* col: the next column number, starting from 1.
	 * it means tha columns 1 to col-1 have been output. I guess. FC.
	 */
	int col, i, line_num = 1;
	unformatted u = sentence_unformatted(s);
	string label = unformatted_label(u);
	int em = unformatted_extra_margin(u);
	int n = unformatted_number(u);
	cons *lw = unformatted_words(u);
	
	/* first 6 columns (0-5)
	 */
	/* 05/08/2003 - Nga Nguyen - Add code for C prettyprinter */

	if (label != (char *) NULL && !string_undefined_p(label)) {
	    /* Keep track of the attachment against the padding: */
	    deal_with_attachments_in_this_string(label,
						 position_in_the_output);
	    /* In C, a label cannot begin with a number so "l" is added for this case*/
	    fprintf_sentence(fd,get_bool_property("PRETTYPRINT_C_CODE")?
			     (isdigit(label[0])?"l%s:":"%s"):"%-5s ", label);
	}
	else {
	  fprintf_sentence(fd,get_bool_property("PRETTYPRINT_C_CODE")?"":"      ");
	}
	
	/* FI: do not indent too much (9 June 1995) */
	em = (em > MAX_START_COLUMN) ? MAX_START_COLUMN : em;
	
	for (i = 0; i < em; i++) 
	    putc_sentence(' ', fd);
	
	col = 7+em;
	
	pips_assert("not too many columns", col <= MAX_END_COLUMN);
	
	MAP(STRING, w, 
	{

	  if (get_bool_property("PRETTYPRINT_C_CODE"))
	    fprintf_sentence(fd, "%s", w);
	  else {
	    
	    /* if the string fits on the current line: no problem */
	    if (col + strlen(w) <= 70) {
		deal_with_attachments_in_this_string(w,
						     position_in_the_output);
		col += fprintf_sentence(fd, "%s", w);
	    }
	    /* if the string fits on one line: 
	     * use the 88 algorithm to break as few
	     * syntactic constructs as possible */
	    else
		if((int)strlen(w) < 70-7-em) {
		    if (col + strlen(w) > 70) {
			/* Complete current line with the statement
                           line number, if it is significative: */
			if (n > 0 &&
			    get_bool_property("PRETTYPRINT_STATEMENT_NUMBER"))
			{
			    for (i = col; i <= MAX_END_COLUMN; i++) 
				putc_sentence(' ', fd);
			    fprintf_sentence(fd, "%04d", n);
			}
			
			/* start a new line with its prefix */
			putc_sentence('\n', fd);

			if(label != (char *) NULL && !string_undefined_p(label)
			   && (strcmp(label,"CDIR$")==0
			       || strcmp(label,"CDIR@")==0
			       || strcmp(label,"CMIC$")==0)) {
			    /* Special label for Cray directives */
			    fprintf_sentence(fd, "%s%d", label, 
					     (++line_num)%10);
			}
			else
			    fprintf_sentence(fd, "     &");

			for (i = 0; i < em; i++)
			    putc_sentence(' ', fd);

			col = 7+em;
		    }
		    deal_with_attachments_in_this_string
			(w, position_in_the_output);
		    col += fprintf_sentence(fd, "%s", w);
		}
	    /* if the string has to be broken in at least two lines: 
	     * new algorithmic part
	     * to avoid line overflow (FI, March 1993) */
		else {
		    char * line = w;
		    int ncar;

		    /* Complete the current line, but not after :-) */
		    ncar = MIN(MAX_END_COLUMN - col + 1, strlen(line));;
		    deal_with_attachments_in_this_string_length
			(line, position_in_the_output, ncar);
		    fprintf_sentence(fd, "%.*s", ncar, line);
		    line += ncar;
		    col = MAX_END_COLUMN;
		    
		    pips_debug(9, "line to print, col=%d\n", col);

		    while(strlen(line)!=0) 
		    {
			ncar = MIN(MAX_END_COLUMN - 7 + 1, strlen(line));

			/* start a new line with its prefix but no indentation
			 * since string constants may be broken onto two lines
			 */
			putc_sentence('\n', fd);

			if(label != (char *) NULL 
			   && (strcmp(label,"CDIR$")==0
			       || strcmp(label,"CDIR@")==0
			       || strcmp(label,"CMIC$")==0)) {
			    /* Special label for Cray directives */
			    (void) fprintf_sentence
				(fd, "%s%d", label, (++line_num)%10);
			}
			else
			    (void) fprintf_sentence(fd, "     &");

			col = 7 ;
			deal_with_attachments_in_this_string_length
			    (line, position_in_the_output, ncar);
			(void) fprintf_sentence(fd, "%.*s", ncar, line);
			line += ncar;
			col += ncar;
		    }
		}
	  }
	},
	    lw);

	pips_debug(9, "line completed, col=%d\n", col);
	pips_assert("not too many columns", col <= MAX_END_COLUMN+1);

	/* Output the statement line number on the right end of the
           line: */
	if (n > 0 && get_bool_property("PRETTYPRINT_STATEMENT_NUMBER")) {
	    for (i = col; i <= MAX_END_COLUMN; i++) 
		putc_sentence(' ', fd);
	    fprintf_sentence(fd, "%04d", n);
	}
	putc_sentence('\n', fd);
    }
}

void 
print_text(FILE *fd, text t)
{
    print_sentence_init();
    MAP(SENTENCE, s, print_sentence(fd, s), text_sentences(t));
}

void 
dump_sentence(sentence s)
{
    print_sentence(stderr, s);
}

/* FI: print_text() should be fprint_text() and dump_text(), print_text() */

void 
dump_text(text t)
{
    print_text(stderr, t);
}

/* Convert a word list into a string and translate the position of
   eventual attachment accordingly: */
string 
words_to_string(list ls)
{
    int size = 1; /* 1 for null termination. */
    string buffer, p;

    /* computes the buffer length.
     */
    MAP(STRING, s, size+=strlen(s), ls);
    buffer = (char*) malloc(sizeof(char)*size);
    pips_assert("malloc ok", buffer);

    /* appends to the buffer...
     */
    buffer[0] = '\0';
    p=buffer;
    MAP(STRING, s, 
	{ strcat_word_and_migrate_attachments(p, s); p+=strlen(s); }, ls);

    return buffer;
}

void 
print_words(FILE * fd, list lw)
{
    string s = words_to_string(lw);
    fputs(s, fd);
    free(s);
}

void 
dump_words(list lw)
{
    print_words(stderr, lw);
}


/********************************************************************* DEBUG */

static void 
debug_word(string w)
{
    fprintf(stderr, "# string--%s--\n", w? w: "<null>");
}

void
debug_words(list /* of string */ l)
{
    gen_map(debug_word, l);
}

static void
debug_formatted(string s)
{
    fprintf(stderr, "# formatted\n%s\n# end formatted\n", s);
}

static void 
debug_unformatted(unformatted u)
{
    fprintf(stderr, "# unformatted\n# label %s, %d, %d\n",
	    unformatted_label(u)? unformatted_label(u): "<null>", 
	    unformatted_number(u), unformatted_extra_margin(u));
    debug_words(unformatted_words(u));
    fprintf(stderr, "# end unformatted\n");
}

void 
debug_sentence(sentence s)
{
    fprintf(stderr, "# sentence\n");
    switch (sentence_tag(s))
    {
    case is_sentence_formatted: 
	debug_formatted(sentence_formatted(s)); 
	break;
    case is_sentence_unformatted: 
	debug_unformatted(sentence_unformatted(s)); 
	break;
    default:
	pips_internal_error("unexpected sentence tag %d\n", sentence_tag(s));
    }
	
    fprintf(stderr,"# end sentence\n");
}

void 
debug_text(text t)
{
    fprintf(stderr, "# text\n");
    gen_map(debug_sentence, text_sentences(t));
    fprintf(stderr,"# end text\n");
    
}
