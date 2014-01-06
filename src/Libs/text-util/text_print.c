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

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text.h"

#include "misc.h"
#include "properties.h"

/* FI: just to make sure that text.h is built; pips-makemake -l does not
   take into account a library whose modules do not use the library header */
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
 * FI: I had to change this module to handle string longer than the
 * space available on one line; I tried to preserve as much as I could
 * of the previous behavior to avoid pseudo-hyphenation at the wrong
 * place and to avoid extensicve problems with validate; the resulting
 * code is lousy, of course; FI, 15 March 1993
 *
 * RK: the print_sentence could print lower case letter according to
 * a property... 17/12/1993.
 */

#define MAX_END_COLUMN_F77      (72)
#define MAX_END_COLUMN_F95     (132)
/* Define a huge line size value because the previous one (999) was too
   small for PIPS generated declarations after the outliner. Do not use
   INT_MAX because there are some assertions such as
   (... < MAX_END_COLUMN_DEFAULT + 1).
   So (INT_MAX/2) is a rough approximation. :-) */
#define MAX_END_COLUMN_DEFAULT (INT_MAX/2)
#define MAX_START_COLUMN 	      (42)
#define C_STATEMENT_LINE_COLUMN (71)
#define C_STATEMENT_LINE_STEP   (15)

void print_sentence(FILE * fd, sentence s) {
  enum language_utype lang = get_prettyprint_language_tag();


  if (sentence_formatted_p(s)) {
    string ps = sentence_formatted(s);

    while(*ps) {
      char c = *ps;
      putc_sentence(c, fd);
      deal_with_attachments_at_this_character(ps, position_in_the_output);
      ps++;
    }
  } else {
    /* col: the next column number, starting from 1.
     * it means tha columns 1 to col-1 have been output. I guess. FC.
     */
    int i, line_num = 1;
    unsigned int col;
    unformatted u = sentence_unformatted(s);
    string label = unformatted_label(u);
    int em = unformatted_extra_margin(u);
    int n = unformatted_number(u);
    cons *lw = unformatted_words(u);
    int max_line_size;
    if(lang==is_language_fortran95) {
      max_line_size = MAX_END_COLUMN_F95;
    } else if(lang==is_language_fortran ) {
      max_line_size =MAX_END_COLUMN_F77;
    } else {
      max_line_size = MAX_END_COLUMN_DEFAULT;
    }

    /* first 6 columns (0-5)
     */
    /* 05/08/2003 - Nga Nguyen - Add code for C prettyprinter */

    if (label != NULL && !string_undefined_p(label)) {
      /* Keep track of the attachment against the padding: */
      deal_with_attachments_in_this_string(label, position_in_the_output);

      switch(lang) {
        case is_language_fortran:
          fprintf_sentence(fd, "%-5s ", label);
          break;
        case is_language_fortran95:
          // Check that the label is non empty
          if(*label!='\0') {
            fprintf_sentence(fd, "%-5s ", label);
          }
          break;
        case is_language_c:
          /* C prettyprinter: a label cannot begin with a number
           * so "l" is added for this case
           */
          if (strlen(label) > 0)
            fprintf_sentence(fd, get_C_label_printf_format(label), label);
          break;
        default:
          pips_internal_error("language unknown not handled");
      }
    } else if (lang==is_language_fortran) {
      fprintf_sentence(fd, "      ");
    }


    /* FI: do not indent too much (9 June 1995) */
    em = (em > MAX_START_COLUMN) ? MAX_START_COLUMN : em;
    /* Initial tabulation, if needed: do not put useless SPACEs
     in output file.
     Well, it's difficult to know if it is useful or not. The
     test below leads to misplaced opening braces.
     */
    if (!ENDP(lw)
    /*&& gen_length(lw)>1
     && !same_string_p(STRING(CAR(lw)), "\n")*/)
      for (i = 0; i < em; i++)
        putc_sentence(' ', fd);

    col = em;

    if (lang == is_language_fortran) {
      col = col + 7; /* Fortran77 start on 7th column */
    }

    pips_assert("not too many columns", col <= max_line_size - 2);
    FOREACH(string, w, lw) {
      switch(lang) {
        case is_language_c:
          col += fprintf_sentence(fd, "%s", w);
          break;
        case is_language_fortran:
        case is_language_fortran95: {
          int max_space_on_a_line = max_line_size - em;
          if(lang==is_language_fortran) {
            max_space_on_a_line -= 7;
          }


          /* if the string fits on the current line: no problem */
          if (col + strlen(w) <= max_line_size - 2) {
            deal_with_attachments_in_this_string(w, position_in_the_output);
            col += fprintf_sentence(fd, "%s", w);
          }
          /* if the string fits on one line:
           * use the 88 algorithm to break as few
           * syntactic constructs as possible */
          else if ((int)strlen(w) < max_space_on_a_line) {
              /* Complete current line with the statement
               line number, if it is significative: */
              if (n > 0 && get_bool_property("PRETTYPRINT_STATEMENT_NUMBER")) {
                for (i = col; i <= max_line_size; i++) {
                  putc_sentence(' ', fd);
                }
                if (lang == is_language_fortran95) {
                  fprintf_sentence(fd, "! %04d", n);
                } else {
                  fprintf_sentence(fd, "%04d", n);
                }
              }

              if(lang==is_language_fortran95) {
                 /* prepare to cut the line */
                 fprintf_sentence(fd," &");
               }

              /* start a new line with its prefix */
              putc_sentence('\n', fd);

              if (label != (char *)NULL && !string_undefined_p(label)
                  && (strcmp(label, "CDIR$") == 0 || strcmp(label, "CDIR@")
                      == 0 || strcmp(label, "CMIC$") == 0)) {
                pips_assert("Cray with F95 not handled",
                            lang!=is_language_fortran95);
                /* Special label for Cray directives */
                fprintf_sentence(fd, "%s%d", label, (++line_num) % 10);
              } else if (lang == is_language_fortran) {
                fprintf_sentence(fd, "     &");
              }

              for (i = 0; i < em; i++)
                putc_sentence(' ', fd);

              if (lang == is_language_fortran) {
                col = 7 + em;
              } else {
                col = em;
              }
            deal_with_attachments_in_this_string(w, position_in_the_output);
            col += fprintf_sentence(fd, "%s", w);
          }
          /* if the string has to be broken in at least two lines:
           * new algorithmic part
           * to avoid line overflow (FI, March 1993) */
          else {
            char * line = w;
            int ncar;

            /* Complete the current line, but not after :-) */
            ncar = MIN(max_line_size - col + 1, strlen(line));
            ;
            deal_with_attachments_in_this_string_length(line,
                                                        position_in_the_output,
                                                        ncar);
            fprintf_sentence(fd, "%.*s", ncar, line);
            line += ncar;
            col = max_line_size;

            pips_debug(9, "line to print, col=%d\n", col);

            while(strlen(line) != 0) {
              ncar = MIN(max_line_size - 7 + 1, strlen(line));


              /* start a new line with its prefix but no indentation
               * since string constants may be broken onto two lines
               */
              if (lang == is_language_fortran95) {
                /* prepare to cut the line */
                fprintf_sentence(fd, " &");
              }
              putc_sentence('\n', fd);

              if (label != (char *)NULL
                  && (strcmp(label, "CDIR$") == 0 || strcmp(label, "CDIR@")
                      == 0 || strcmp(label, "CMIC$") == 0)) {
                /* Special label for Cray directives */
                (void)fprintf_sentence(fd, "%s%d", label, (++line_num) % 10);
              } else if (lang == is_language_fortran) {
                (void)fprintf_sentence(fd, "     &");
                col = 7;
              } else {
                col = 0;
              }
              deal_with_attachments_in_this_string_length(line,
                                                          position_in_the_output,
                                                          ncar);
              (void)fprintf_sentence(fd, "%.*s", ncar, line);
              line += ncar;
              col += ncar;
            }
          }
          break;
        }
        default:
          pips_internal_error("Language unknown !");
          break;
      }
    }

    pips_debug(9, "line completed, col=%d\n", col);
    pips_assert("not too many columns", col <= max_line_size + 1);

    /* statement line number starts at different column depending on
     * the used language : C or fortran
     */
    size_t column_start = 0;
    switch (get_prettyprint_language_tag()) {
      case is_language_fortran:
        /* fortran case right the line number on the right where characters
         are ignored by a f77 parser*/
        column_start = max_line_size;
        break;
      case is_language_c:
      case is_language_fortran95:
        /* C and F95 case, try to align the line number on the right using
         * commentaries. The alignment is done modulo C_STATEMENT_LINE_STEP
         */
        column_start = C_STATEMENT_LINE_COLUMN;
        while(column_start <= col)
          column_start += C_STATEMENT_LINE_STEP;
        break;
      default:
        pips_internal_error("Language unknown !");
        break;
    }

    /* Output the statement line number on the right end of the
     line: */
    if (n > 0 && get_bool_property("PRETTYPRINT_STATEMENT_NUMBER")) {
      for (size_t i = col; i <= column_start; i++) {
        putc_sentence(' ', fd);
      }
      switch (get_prettyprint_language_tag()) {
        case is_language_fortran:
          fprintf_sentence(fd, "%04d", n);
          break;
        case is_language_c:
          fprintf_sentence(fd, "/*%04d*/", n);
          break;
        case is_language_fortran95:
          fprintf_sentence(fd, "! %04d", n);
          break;
        default:
          pips_internal_error("Language unknown !");
          break;
      }
    }
    putc_sentence('\n', fd);
  }
}

void print_text(FILE *fd, text t)
{
    print_sentence_init();
    MAP(SENTENCE, s, print_sentence(fd, s), text_sentences(t));
}

void dump_sentence(sentence s)
{
    print_sentence(stderr, s);
}

/* FI: print_text() should be fprint_text() and dump_text(), print_text() */

void dump_text(text t)
{
    print_text(stderr, t);
}

string words_join(list ls,const char* sep)
{
    int size = 1; /* 1 for null termination. */
    size_t sep_sz = strlen(sep);
    string buffer, p;

    /* computes the buffer length.
     */
    MAP(STRING, s, size+=strlen(s)+sep_sz, ls);
    buffer = (char*) malloc(sizeof(char)*size);
    pips_assert("malloc ok", buffer);

    /* appends to the buffer...
     */
    buffer[0] = '\0';
    p=buffer;
    FOREACH(STRING, s,ls) {
        strcat_word_and_migrate_attachments(p, s);
        strcat_word_and_migrate_attachments(p, sep);
        p+=strlen(s);
        p+=sep_sz;
    }
    return buffer;
}

/* Convert a word list into a string and translate the position of
   eventual attachment accordingly: */
string words_to_string(list ls)
{
    return words_join(ls,"");
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

void print_words(FILE * fd, list lw)
{
    string s = words_to_string(lw);
    fputs(s, fd);
    free(s);
}

void dump_words(list lw)
{
    print_words(stderr, lw);
}


/********************************************************************* DEBUG */

static void debug_word(string w)
{
    fprintf(stderr, "# string--%s--\n", w? w: "<null>");
}

void
debug_words(list /* of string */ l)
{
    gen_map((gen_iter_func_t)debug_word, l);
}

static void
debug_formatted(string s)
{
    fprintf(stderr, "# formatted\n%s\n# end formatted\n", s);
}

static void debug_unformatted(unformatted u)
{
    fprintf(stderr, "# unformatted\n# label %s, %td, %td\n",
	    unformatted_label(u)? unformatted_label(u): "<null>",
	    unformatted_number(u), unformatted_extra_margin(u));
    debug_words(unformatted_words(u));
    fprintf(stderr, "# end unformatted\n");
}

void debug_sentence(sentence s)
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
	pips_internal_error("unexpected sentence tag %d", sentence_tag(s));
    }

    fprintf(stderr,"# end sentence\n");
}

void debug_text(text t)
{
    fprintf(stderr, "# text\n");
    gen_map((gen_iter_func_t)debug_sentence, text_sentences(t));
    fprintf(stderr,"# end text\n");
}
