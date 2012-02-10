/* $Id$ */

/******************** LEXICAL ANALYZER **************************

  Here are the lexical rules, based on the work of people from
  Open Source Quality projects (http://osq.cs.berkeley.edu/), used
  by the CCured source-to-source translator for C


*****************************************************************/

/*(*
 *
 * Copyright (c) 2001-2003,
 *  George C. Necula    <necula@cs.berkeley.edu>
 *  Scott McPeak        <smcpeak@cs.berkeley.edu>
 *  Wes Weimer          <weimer@cs.berkeley.edu>
 *  Ben Liblit          <liblit@cs.berkeley.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. The names of the contributors may not be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *)
(* FrontC -- lexical analyzer
**
** 1.0	3.22.99	Hugues Cassé	First version.
** 2.0  George Necula 12/12/00: Many extensions
*)*/

%option nounput
%option noinput

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "c_syntax.h"
#include "cyacc.h"

#define FIRST_C_LINE_NUMBER (1)
#define UNDEFINED_C_LINE_NUMBER (-1)


/* To track the user line number, that is the one in the original user file */
static int C_line_number = UNDEFINED_C_LINE_NUMBER;
/* Use a stack to survive to file inclusions if any.

   I guess that it is currently not used since file inclusion is already
   done by the preprocessor... RK
*/
DEFINE_LOCAL_STACK(line_number, _int)

/* To track the absolute line number in the file that is parsed
   use the default flex yylineno that is renamed c_lineno here */
/* Use a stack to survive to file inclusions if any */
DEFINE_LOCAL_STACK(absolute_line_number, _int)

/* Count the number of c in s. */
unsigned int character_occurences_in_string(string s, char c)
{
  string p = string_undefined;
  unsigned int count = 0;

  for(p=s; *p!='\0'; p++) {
    count += (*p==c)? 1 : 0 ;
  }
  return count;
}

int get_current_C_line_number()
{
  return C_line_number;
}

void set_current_C_line_number()
{
  if (C_line_number == UNDEFINED_C_LINE_NUMBER) {
    /* Initialize the user line number... */
    C_line_number = FIRST_C_LINE_NUMBER;
    /* ... and the absolute line number in the current file */
    c_lineno = FIRST_C_LINE_NUMBER;
  }
  else
   pips_internal_error("C_line_number not resetted\n");

  /* Some check on it first? It should have been disallocated by reset_current_C_line_number() */
  make_line_number_stack();
  make_absolute_line_number_stack();
}

/* The line number stack, designed for structured control structure, is
   not used yet. */
void push_current_C_line_number()
{
  int ln = get_current_C_line_number();

  line_number_push(ln);
  absolute_line_number_push(c_lineno);
}

int pop_current_C_line_number()
{
  int ln = line_number_pop();
  c_lineno = absolute_line_number_pop();

  return ln;
}

void reset_current_C_line_number()
{
  C_line_number = UNDEFINED_C_LINE_NUMBER;
  c_lineno = FIRST_C_LINE_NUMBER;

  if(!line_number_empty_p()) {
    pips_internal_error("Line number stack is not empty\n");
  }
  free_line_number_stack();
  free_absolute_line_number_stack();
}

void error_reset_current_C_line_number()
{
  C_line_number = UNDEFINED_C_LINE_NUMBER;
  c_lineno = FIRST_C_LINE_NUMBER;

  free_line_number_stack();
  free_absolute_line_number_stack();
}

/* Comment management:

   - comments within declarations are misplaced (because there is no
     corresponding attachment point)

   - comments for structured control structures such as "for", "switch",
     "while",... are stacked so that we can get them back when building
     the statement at the end of the statement;

     "do .... while()" is not handled properly; "else" cannot carry a
     comment, because the "while" and "else" do not really exist in the
     RI;

   - the comments for other statements are the current comments

   - end-of-line comments placed after a statement are stored as comment
     for the next statement

   - end-of-line before comments that can be retained by the C parser are
     gathered with the comment

   - comments appearing by the end of a block are lost (no NOP statement
     is generated to carry them yet)

   - linefeed and comments inside a statement are collected as a comment
     before statement (some weird impact on print-out may occur); same as
     for declaration statements, but less frequent

   - some comments are still ignored, although available (to be implemented)
*/

static string C_current_comment = string_undefined;

DEFINE_LOCAL_STACK(comments, string)

/* Return the current comment as a string to be freed by the caller and
   reset the current comment. If the current comment is undefined, returns
   a copy of the empty string, "".

   Reset also the current comment.
 */
string get_current_C_comment()
{
  string cc = C_current_comment;
  C_current_comment = string_undefined;
  if (cc != string_undefined) {
    if (cc[0] == '\n') {
      /* If the comment begins with a new-line, it is indeed the new-line
	 that ends the previous statement, so skip it. Quicker than strlen()
	 + memmove(): */
      char * p = &cc[0];
      do {
	p[0] = p[1];
      }
      while (*p++ != '\0');
      /* Note there won't be a memory leak since the orginal '\0' is in the
	 malloc() bloc to be free()ed anyway... */
    }
    /* If the comment is only an empty one, do not retain it: */
    if (cc[0] == '\0') {
      /* Do retain it to keep the statement data structure easy to use, allowing strdup() on its text fields */
      //free(cc);
      //cc = string_undefined;
      ;
    }
    else {
      /* Remove the trailing new-line if any since the RI is already
	 line-oriented at the comment level: This is already done above. */
      char * last_newline = strrchr(cc, '\n');
      if (last_newline != NULL && last_newline[1] == '\0') {
	/* It is a trailing new-line: just get rid of it: */
	//last_newline[0] = '\0';
	;
      }
    }
  }
  else
    cc = strdup("");
  /* pips_debug(3, "get_current_C_comment comment \"%s\"\n",
     cc); */
  return cc;
}

/* Push the current C comment so that we can get it back when building the
   statement later

   This reset the current comment through get_current_C_comment()
*/
void push_current_C_comment()
{
  string cc = get_current_C_comment();

  if (string_undefined_p(cc))
    pips_debug(3, "empty comment pushed at line %d\n",
	       get_current_C_line_number());
  else
    pips_debug(3, "comment \"%s\" pushed at line %d\n", cc,
	       get_current_C_line_number());
   comments_push(cc);
}

/* Pop the current comment.

   @return the previous current comment.

   This is typically used at the end of a statement to be built.

   Note this do not set the current comment. Strange API...
*/
string pop_current_C_comment()
{
  string cc = comments_pop();
  if (string_undefined_p(cc))
    pips_debug(3, "empty comment popped at line %d\n",
	       get_current_C_line_number());
  else
    pips_debug(3, "comment \"%s\" popped at line %d\n", cc,
	       get_current_C_line_number());
  return cc;
}

/* Add a comment to the current one */
void
update_C_comment(string a_comment)
{
  string new_comment = strdup(concatenate(C_current_comment, a_comment, NULL));
  if (!string_undefined_p(C_current_comment))
    free(C_current_comment);
  C_current_comment = new_comment;

  pips_debug(3,"update_C_comment %s\n", C_current_comment);
}

/* Discard a C comment because we don't know how to deal with them */
void discard_C_comment()
{
  if(!string_undefined_p(C_current_comment)) {
    if (character_occurences_in_string(C_current_comment, '\n')
	== strlen(C_current_comment)) {
      /* The comments are only made of '\n', just silently discarding them */
      pips_debug(3,"The \\n are lost, so the code presentation may be wrong...\n");
    }
    else {
      /*
      pips_user_warning("Comment \"%s\" is lost at line %d, "
			"probably because comments cannot be attached to declarations.\n",
			C_current_comment, C_line_number);
      */
      pips_debug(8, "Comment \"%s\" is lost at line %d, "
		 "probably because comments cannot be attached to declarations.\n",
		 C_current_comment, C_line_number);
    }
    free(C_current_comment);
    C_current_comment = string_undefined;
  }
}

/* reset and reset_error should be handled differently */
void reset_C_comment(bool is_compilation_unit_p)
{
  if(!string_undefined_p(C_current_comment)) {
    free(C_current_comment);
    C_current_comment = string_undefined;
  }
  /* Comments in the compilation unit are lost because they are related only to
     declarations and because comments on declarations are lost. Also, comments
     located at the end of a block are lost, as we do not generate an extra NOP to carry them. */
  if(!is_compilation_unit_p && !comments_empty_p()) {
    int count = 0;
    pips_user_warning("Comments stack is not empty (only meaningful comments are shown):\n");
    while(!comments_empty_p()) {
      string c = comments_pop();
      count++;
      if(strcmp(c, "\n")!=0)
        fprintf(stderr, "Element %d: \"%s\"\n", count, c);
      free(c);
    }
    /* pips_internal_error("Comments stack is not empty\n"); */
  }
  clear_C_comment();
  free_comments_stack();
}

void error_reset_C_comment(bool is_compilation_unit_p __attribute__ ((__unused__)))
{
  if(!string_undefined_p(C_current_comment)) {
    free(C_current_comment);
    C_current_comment = string_undefined;
  }
  clear_C_comment();
  free_comments_stack();
}

void clear_C_comment()
{
  if(!string_undefined_p(C_current_comment)) {
    free(C_current_comment);
    C_current_comment = string_undefined;
  }
  /* Comments in the compilation unit and outside of function bodies
     are lost because they are related only to
     declarations and because comments on declarations are lost.*/
  if(!comments_empty_p()) {
    int count = 0;
    pips_debug(3, "Comments stack is not empty:\n");
    while(!comments_empty_p()) {
      string c = comments_pop();
      count++;
      pips_debug(3, "Element %d: \"%s\"\n",
		 count, string_undefined_p(c) ? "string undefined" : c);
      if(!string_undefined_p(c))
        free(c);
    }
  }
  pips_assert("The comment stack is empty\n", comments_empty_p());
}

void init_C_comment()
{
  if(!string_undefined_p(C_current_comment)) {
    pips_internal_error("Missing reset for C_current_comment");
  }
  if(!stack_undefined_p(comments_stack) && !STACK_NULL_P(comments_stack) && !comments_empty_p()) {
    pips_internal_error("Comment stack is not empty");
  }
  make_comments_stack();
}
/* compatibility layer for bsd */
static void bsd_rewrite(char ** pyytext) {
	char* aliases [][2] = {
		{ "__stdinp", "stdin" },
		{ "__stdoutp", "stdout" },
		{ "__stderrp", "stderr" },
		{ "__isnanl", "isnanl" },
		{ "__isnanf", "isnanf" },
		{ "__isnan", "isnan" },
		{ NULL, NULL }
	};
	for(char *(*iter)[2] = &aliases[0] ; (*iter)[0] ; ++iter) {
		if(same_string_p(*pyytext, (*iter)[0] )) {
			*pyytext=(*iter)[1];
			break;
		}
	}
}

%}


/* This should be used instead of the manual computation of
   C_line_number...
*/
%option   yylineno

 /* To deal more cleanly with the comments:
    %x COMMENT
 */
 /* To deal cleanly with complex pragma line escape, use a separate state: */
%x the_end_of_pragma

decdigit [0-9]
octdigit [0-7]
hexdigit [0-9a-fA-F]
letter [a-zA-Z]
usuffix [uU]
lsuffix (l|L|ll|LL)
intsuffix (({lsuffix})|({usuffix})|({usuffix}{lsuffix})|({lsuffix}{usuffix}))
hexprefix (0[xX])
intnum ({decdigit}+{intsuffix}?)
octnum (0{octdigit}+{intsuffix}?)
hexnum ({hexprefix}{hexdigit}+{intsuffix}?)
exponent ([eE][\+\-]?{decdigit}+)
fraction (\.{decdigit}+)
decfloat (({intnum}?{fraction})|({intnum}{exponent})|({intnum}?{fraction}{exponent})|({intnum}\.)|({intnum}\.{exponent}))
hexfraction (({hexdigit}*"."{hexdigit}+)|({hexdigit}+))
binexponent ([pP][\+\-]?{decdigit}+)
floatonlynum ({decfloat}{floatsuffix}?)
floatexponent ([pP][\+\-]?{floatonlynum})
hexfloat (({hexprefix}{hexfraction}{binexponent})|({hexprefix}{hexdigit}+{binexponent})|({hexprefix}{hexdigit}{floatexponent}))
floatsuffix [fFlL]
complexsuffix [ij]
floatnum (({decfloat}|{hexfloat}){floatsuffix}?)
complexnum (({decfloat}|{hexfloat})({complexsuffix}{floatsuffix}?|{floatsuffix}{complexsuffix}))
ident ({letter}|"_")({letter}|{decdigit}|"_")*
attribident (({letter}|"_")({letter}|{decdigit}|"_"|":"))
escape \\
 /* From C norm A.1: */
hex_escape ({escape}x{hexdigit}{1,2})
wide_hex_escape ({escape}x{hexdigit}{1,8})
oct_escape ({escape}{octdigit}{1,3})
char_escape ({escape}[abfnrtv\'\\?"])
/* " */
universal-character-name ({escape}(u{hexdigit}{4}|U{hexdigit}{8}))


%%

<the_end_of_pragma>("\\\n"|[^\n])*"\n" {
		 /* Parse some potentially escaped line up to a normal new
			 line that is the end of a pragma: */
		 /* Return the pragma value without the final \n: */
		 c_lval.string = strndup(yytext, strlen(yytext) - 1);
		 BEGIN(INITIAL);
		 return(TK_PRAGMA);
	       }


("/*"([^*]|("*"+[^/*]))*"*"+"/")|("//"("\\\n"|[^\n])*"\n"?)	{ //  */
                                pips_debug(9,"Comment %s\n",yytext);
                                C_line_number += character_occurences_in_string(yytext, '\n');
                                update_C_comment(yytext);
                        }

^"#"("\\\n"|[ \t])*"pragma"([ \t])* {
                /* Look for a # followed by any whitespace and any
		   backslashed end-of line, followed by "pragma" up to the
		   end of line, dealing with any backslashed new lines: */
		 pips_debug(1, "#pragma found: \"%s\"\n", yytext);
		 /* To deal with complex line escape, put the end of the
		    analysis in another rule: */
		 BEGIN(the_end_of_pragma);
		 }

^"#"[ \t]*{decdigit}+[^\n]*"\n"             {
                                // \n is not part of the pragma
                                // The line count is correct but empty lines
                                // seem to be inserted at the beginning of
                                // the compilation unit files
                                pips_debug(9,"Pragma-like comment %s\n",yytext);
                                /* the preprocessor seems to provide line information in pragmas
                                   which are not #line pragmas. */
                                if(/* strstr(yytext, "#line")==yytext && */ strlen(yytext)>=4) {
                                   int initial_C_line_number = -1;
                                   int items = sscanf(yytext+1, "%d", &initial_C_line_number);
                                   if(items==1) {
                                      C_line_number = initial_C_line_number;
                                   }
                                   else {
                                      pips_user_warning("No line number after a #: \"%s\".\n", yytext);
                                      CParserError("Ill. formated # pragma\n");
                                   }
                                }
                        }

\n                      {
                                pips_debug(9,"New line %d\n", C_line_number);

                                /* Gather a free "\n" as a comment to keep
				   user presentation information. If it is
				   in a place where we can not store a
				   comment in the RI, it will be silenty
				   lost. */
				update_C_comment(yytext);
				pips_debug(3,"CNew line %s\n", yytext);
                                C_line_number++;
                        }

[[:blank:]]		{
                                /* Eat up whitespaces */
                        }

"_Pragma" 	        {
                                pips_debug(9,"_Pragma found in \"%s\"\n",yytext);
				return(TK__Pragma);
                        }

\'([^\\]|{char_escape}|{oct_escape}|{hex_escape}|{universal-character-name})\'           {
                                /* Escaped character constants. Their
                                   syntax is understood in PIPS character
                                   constant construtors. */
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_CHARCON);
                        }

L\'([^\\]|{char_escape}|{oct_escape}|{wide_hex_escape}|{universal-character-name})\'           {
                                /* Escaped wide character constants. Their
                                   syntax is understood in PIPS character
                                   constant construtors. */
                                pips_debug(9,"TK_CHARCON wide character constant: %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_CHARCON);
                        }

\"(\\\"|[^\"\\]|\\[^\"])*\"	{
                                pips_debug(9,"TK_STRINGCON regular or wide string: %s\n",yytext);
                                /* the "world" in L"Hello, " "world"
                                   should be treated as wide even though
                                   there's no L immediately preceding
                                   it */
                                /* Indeed there is no special caracter
				   handling here since it is done
				   somewhere else in PIPS in the constant
				   string constructor. */
                                c_lval.string = strdup(yytext);
				return(TK_STRINGCON);
                        }

L\"(\\\"|[^\"\\]|\\[^\"])*\"	{
                                pips_debug(9,"TK_WSTRINGCON wide string: %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_WSTRINGCON);
                        }
{floatnum}		{
                                pips_debug(9,"TK_FLOATCON %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_FLOATCON);
                        }
{complexnum}		{
                                pips_debug(5,"TK_COMPLEXCON %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_COMPLEXCON);
                        }
{hexnum}		{
                                pips_debug(9,"Hexnum TK_INTCON %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{octnum}		{
                                pips_debug(9,"Octnum TK_INTCON %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{intnum}		{
                                pips_debug(9,"TK_INTCON %s\n",yytext);
                                c_lval.string = strdup(yytext);
				return(TK_INTCON);
                        }
"!quit!"		{
				pips_debug(9,"TK_EOF %s\n",yytext);
				yyterminate();
			}
"..."			{
				pips_debug(9,"TK_ELLIPSIS %s\n",yytext);
				return(TK_ELLIPSIS);
			}
"+="			{
				pips_debug(9,"TK_PLUS_EQ %s\n",yytext);
				return(TK_PLUS_EQ);
			}
"-="			{
				pips_debug(9,"TK_MINUS_EQ %s\n",yytext);
				return(TK_MINUS_EQ);
			}
"*="			{
				pips_debug(9,"TK_STAR_EQ %s\n",yytext);
				return(TK_STAR_EQ);
			}
"/="			{
				pips_debug(9,"TK_SLASH_EQ %s\n",yytext);
				return(TK_SLASH_EQ);
			}
"%="			{
				pips_debug(9,"TK_PERCENT_EQ %s\n",yytext);
				return(TK_PERCENT_EQ);
			}
"|="			{
				pips_debug(9,"TK_PIPE_EQ %s\n",yytext);
				return(TK_PIPE_EQ);
			}
"&="			{
				pips_debug(9,"TK_AND_EQ %s\n",yytext);
				return(TK_AND_EQ);
			}
"^="			{
				pips_debug(9,"TK_CIRC_EQ %s\n",yytext);
				return(TK_CIRC_EQ);
			}
"<<="			{
				pips_debug(9,"TK_INF_INF_EQ %s\n",yytext);
				return(TK_INF_INF_EQ);
			}
">>="			{
				pips_debug(9,"TK_SUP_SUP_EQ %s\n",yytext);
				return(TK_SUP_SUP_EQ);
			}
"<<"			{
				pips_debug(9,"TK_INF_INF %s\n",yytext);
				return(TK_INF_INF);
			}
">>"			{
				pips_debug(9,"TK_SUP_SUP %s\n",yytext);
				return(TK_SUP_SUP);
			}
"=="			{
				pips_debug(9,"TK_EQ_EQ %s\n",yytext);
				return(TK_EQ_EQ);
			}
"!="			{
				pips_debug(9,"TK_EXCLAM_EQ %s\n",yytext);
				return(TK_EXCLAM_EQ);
			}
"<="			{
				pips_debug(9,"TK_INF_EQ %s\n",yytext);
				return(TK_INF_EQ);
			}
">="			{
				pips_debug(9,"TK_SUP_EQ %s\n",yytext);
				return(TK_SUP_EQ);
			}
"="			{
				pips_debug(9,"TK_EQ %s\n",yytext);
				return(TK_EQ);
			}
"<"			{
				pips_debug(9,"TK_INF %s\n",yytext);
				return(TK_INF);
			}
">"			{
				pips_debug(9,"TK_SUP %s\n",yytext);
				return(TK_SUP);
			}
"++"			{
				pips_debug(9,"TK_PLUS_PLUS %s\n",yytext);
				return(TK_PLUS_PLUS);
			}
"--"			{
				pips_debug(9,"TK_MINUS_MINUS %s\n",yytext);
				return(TK_MINUS_MINUS);
			}
"->"			{
				pips_debug(9,"TK_ARROW %s\n",yytext);
				return(TK_ARROW);
			}
"+"			{
				pips_debug(9,"TK_PLUS %s\n",yytext);
				return(TK_PLUS);
			}
"-"			{
				pips_debug(9,"TK_MINUS %s\n",yytext);
				return(TK_MINUS);
			}
"*"			{
				pips_debug(9,"TK_STAR %s\n",yytext);
				return(TK_STAR);
			}
"/"			{
				pips_debug(9,"TK_SLASH %s\n",yytext);
				return(TK_SLASH);
			}
"%"			{
				pips_debug(9,"TK_PERCENT %s\n",yytext);
				return(TK_PERCENT);
			}
"!"			{
				pips_debug(9,"TK_EXCLAM %s\n",yytext);
				return(TK_EXCLAM);
			}
"&&"			{
				pips_debug(9,"TK_AND_AND %s\n",yytext);
				return(TK_AND_AND);
			}
"||"			{
				pips_debug(9,"TK_PIPE_PIPE %s\n",yytext);
				return(TK_PIPE_PIPE);
			}
"&"			{
				pips_debug(9,"TK_AND %s\n",yytext);
				return(TK_AND);
			}
"|"			{
				pips_debug(9,"TK_PIPE %s\n",yytext);
				return(TK_PIPE);
			}
"^"			{
				pips_debug(9,"TK_CIRC %s\n",yytext);
				return(TK_CIRC);
			}
"?"			{
				pips_debug(9,"TK_QUEST %s\n",yytext);
				return(TK_QUEST);
			}
":"			{
				pips_debug(9,"TK_COLON %s\n",yytext);
				return(TK_COLON);
			}
"~"			{
				pips_debug(9,"TK_TILDE %s\n",yytext);
				return(TK_TILDE);
			}
"\{"                    {
				pips_debug(9,"TK_LBRACE %s\n",yytext);
                                // We do not know what to do for a block,
                                // but we would like to keep them for struct
                                // and union declaration
                                //discard_C_comment();
				return(TK_LBRACE);
			}
"\}"			{
				pips_debug(9,"TK_RBRACE %s\n",yytext);
                                //discard_C_comment();
				return(TK_RBRACE);
			}
"["			{
				pips_debug(9,"TK_LBRACKET %s\n",yytext);
				return(TK_LBRACKET);
			}
"]"			{
				pips_debug(9,"TK_RBRACKET %s\n",yytext);
				return(TK_RBRACKET);
			}
"("			{
				pips_debug(9,"TK_LPAREN %s\n",yytext);
				return(TK_LPAREN);
			}
")"			{
				pips_debug(9,"TK_RPAREN %s\n",yytext);
				return(TK_RPAREN);
			}
";"			{
				pips_debug(9,"TK_SEMICOLON %s\n",yytext);
                                /* discard_C_comment(); */
				return(TK_SEMICOLON);
			}
","			{
				pips_debug(9,"TK_COMMA %s\n",yytext);
				return(TK_COMMA);
			}
"."			{
				pips_debug(9,"TK_DOT %s\n",yytext);
				return(TK_DOT);
			}
"sizeof"		{
				pips_debug(9,"TK_SIZEOF %s\n",yytext);
				return(TK_SIZEOF);
			}
"__asm"                 {
				pips_debug(9,"TK_ASM %s\n",yytext);
				return(TK_ASM);
			}
"__va_list" 	        {
	/* convert __va_list from bsd into __builtin_va_list */
	pips_debug(5,"TK_VA_LIST %s\n",yytext);
	yytext = strdup("__builtin_va_list");
	int t = is_c_keyword_typedef(yytext);
	pips_assert("builtin valist found\n",t>0);
	pips_debug(9,"Keyword or typedef name: %s\n",yytext);
	c_lval.string = strdup(yytext);
	return (t);
}

"__extension__"         {
                                pips_user_warning("gcc extension keyword \"__extension__\" is ignored\n");
                        }
"__inline"              {
                                pips_user_warning("gcc extension keyword \"__inline\" is ignored\n");
                        }
"__inline__"            {
                                pips_user_warning("gcc extension keyword \"__inline__\" is ignored\n");
                        }
"__signed__"            {
                                pips_user_warning("gcc extension keyword \"__signed__\" is ignored\n");
                        }
{ident}			{
				bsd_rewrite(&yytext);
                /* C keywords or identifiers */
				int t = is_c_keyword_typedef(yytext);
                                if (t>0)
				   {
				       pips_debug(9,"Keyword or typedef name: %s\n",yytext);
                                       c_lval.string = strdup(yytext);
				       /* Some comments are lost, but not especially in declarations
				       because type keywords may appear in casts */
				       if(t==TK_ELSE /*|| (t>=TK_CHAR && t <=TK_REGISTER)*/ )
				         discard_C_comment();
				       return (t);
				   }
				else
				   {
				       pips_debug(9,"Identifier: %s\n",yytext);
                                       c_lval.string = strdup(yytext);
                                       /* push_current_C_comment(); */
				       return(TK_IDENT);
				   }
			}
<<EOF>>			{
				pips_debug(9,"TK_EOF %s\n",yytext);
				return(TK_EOF);
			}
.			{
			        pips_user_warning("Unrecognized character %s\n",yytext);
			}
%%

/* The default function called when a parsing error happens */
void c_error(char * msg)
{
  user_warning("yyerror", "C %s near \"%s\" at user line %d (local line %d in the effective file)\n", msg, yytext, get_current_C_line_number(), c_lineno);
  CParserError("Parsing failed\n");
}


/* Mainly reset the line number and restart on c_in... */
void c_reset_lex(void)
{
    c_lineno = 1;
#ifdef FLEX_SCANNER
    c_restart(c_in);
#else
    /* reset lex... Might be better to read the whole file?
     */
     /* Huh... What is this ? RK */
    syn_sptr = syn_sbuf;
# define MMNEWLINE 10
    syn_previous = MMNEWLINE;
#endif
}

#ifndef c_wrap
int c_wrap() { return 1;}
#endif
