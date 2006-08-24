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

%{
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

static int C_line_number = UNDEFINED_C_LINE_NUMBER;

DEFINE_LOCAL_STACK(line_number, int)

int get_current_C_line_number()
{
  return C_line_number;
}

void set_current_C_line_number()
{
  if(C_line_number==UNDEFINED_C_LINE_NUMBER)
    C_line_number = FIRST_C_LINE_NUMBER;
  else
    pips_internal_error("C_line_number not resetted\n");

  /* Some check on it first? It should have been disallocated by reset_current_C_line_number() */
  make_line_number_stack();
}

/* The line number stack, designed for structured control structure, is not used yet. */

void push_current_C_line_number()
{
  int ln = get_current_C_line_number();

  line_number_push(ln);
}

int pop_current_C_line_number()
{
  int ln = line_number_pop();

  return ln;
}

void reset_current_C_line_number()
{
  C_line_number = UNDEFINED_C_LINE_NUMBER;

  if(!line_number_empty_p()) {
    pips_internal_error("Line number stack is not empty\n");
  }
  free_line_number_stack();
}

/* Comment management:

   - comments in declarations are lost (because there is no
corresponding attachment point)

   - comments for structured control structures such as "for", "switch", "while",... are stacked;
     "do .... while()" is not handled properly; "else" cannot carry a comment;

   - the comments for other statements are the current comments

   - end-of-line comments placed after a statement are stored as
comment for the next statement

   - comments appearing by the end of a block are lost (no NOP
statement is generated to carry them)

   - linefeed and comments inside a statement are collected as a before statement
(weird impact on print-out);

   - some comments are still ignored, although available (to be implemented)

*/

static string C_current_comment = string_undefined;

DEFINE_LOCAL_STACK(comments, string)

string get_current_C_comment()
{
  string cc = empty_comments;

  if(!string_undefined_p(C_current_comment)) {
    if(*C_current_comment=='\n')
      cc = strdup(C_current_comment);
    else
      cc = strdup(C_current_comment);
    free(C_current_comment);
    C_current_comment = string_undefined;
    if(*cc=='\000') {
      free(cc);
      cc = empty_comments;
    }
  }

  return cc;
}

void push_current_C_comment()
{
  string cc = get_current_C_comment();

  if(!string_undefined_p(cc)) {
    pips_debug(8, "comment \"%s\" pushed at line %d\n", cc, get_current_C_line_number());
    comments_push(cc);
  }
}

string pop_current_C_comment()
{
  string cc = comments_pop();

  return cc;
}

string update_C_comment(string cc, string nc)
{
  if(string_undefined_p(cc)) {
    /* We duplicate too many LF */
    /* if(strcmp(nc, "\n")!=0) { */
      cc = strdup(nc);
    /* } */
  }
  else {
    string oc = cc;
    cc = strdup(concatenate(cc, nc, NULL));
    free(oc);
  }
  return cc;
}

void discard_C_comment()
{
  if(!string_undefined_p(C_current_comment)) {
    pips_user_warning("Comment \"%s\" is lost at line %d, "
		      "probably because comments cannot be attached to declarations.\n",
		      C_current_comment, C_line_number);
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
    pips_debug(8, "Comments stack is not empty:\n");
    while(!comments_empty_p()) {
      string c = comments_pop();
      count++;
      pips_debug(8, "Element %d: \"%s\"\n", count, c);
      free(c);
    }
  }
  pips_assert("The comment stack is empty\n", comments_empty_p());
}

void set_C_comment()
{
  if(!string_undefined_p(C_current_comment)) {
    pips_internal_error("Missing reset for C_current_comment");
  }
  if(!stack_undefined_p(comments_stack) && !STACK_NULL_P(comments_stack) && !comments_empty_p()) {
    pips_internal_error("Comment stack is not empty");
  }
  make_comments_stack();
}

int character_occurences_in_string(string s, char c)
{
  string p = string_undefined;
  int count = 0;

  for(p=s; *p!='\0'; p++) {
    count += (*p==c)? 1 : 0 ;
  }
  return count;
}

%}

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
binexponent ([pP]["+""-"]?{decdigit}+)
hexfloat (({hexprefix}{hexfraction}{binexponent})|({hexprefix}{hexdigit}+{binexponent}))
floatsuffix [fFlL]
floatnum (({decfloat}|{hexfloat}){floatsuffix}?)
ident ({letter}|"_")({letter}|{decdigit}|"_")* 
attribident (({letter}|"_")({letter}|{decdigit}|"_"|":"))
blank [ \t\r]+
escape ('\\'_)
hex_escape ('\\'[xX]{hexdigit}+)
oct_escape ('\\'{octdigit}{octdigit}?{octdigit}?) 

%%
"/*"([^\*]|(\*+[^\/\*]))*(\*)+"/"	{ 
                                pips_debug(9,"Comment %s\n",yytext);
                                C_line_number += character_occurences_in_string(yytext, '\n');
                                C_current_comment = update_C_comment(C_current_comment, yytext);
                        }
"//"[^\n]*              { 
                                pips_debug(9,"Comment %s\n",yytext);
                                C_current_comment = update_C_comment(C_current_comment, yytext);
                        }
^"#"[^\n]*              { 
                                pips_debug(9,"Pragma comment %s\n",yytext);
                                /* Is it a #line pragma? It has to be after preprocessing */
                                if(/* strstr(yytext, "#line")==yytext && */ strlen(yytext)>=4) {
                                   int initial_C_line_number = -1;
                                   int items = sscanf(yytext+1, "%d", &initial_C_line_number);
                                   if(items==1) {
                                      /* Get rid of the pragma LF itslef */
                                      C_line_number = initial_C_line_number-1;
                                   }
                                   else {
                                      pips_user_warning("No line number in # pragma: \"%s\".\n", yytext);
                                      CParserError("Ill. formated # pragma\n");
                                   }
                                }
                        }
{blank}			{ 
                                /* Eat up whitespaces */
                        }
\n                      { 
                                pips_debug(9,"New line %d\n", C_line_number);

                                /* If in comment area (not checked...), append to current comment */
                                C_current_comment = update_C_comment(C_current_comment, yytext);
                                C_line_number++;
                        }

"_Pragma" 	        { 
                                pips_debug(9,"Pragma not implemented %s\n",yytext);
				return(TK_PRAGMA);
                        }

\'[^\\]\'               { 
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
\'\\[abfnrtv\'\\]\'     { 
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
\'\\0[0-9]*\'           { 
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
L\'[^\\]\'              { 
                                pips_debug(9,"TK_CHARCON wide character constant: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
L\'\\[abfnrtv\'\\]\'    { 
                                pips_debug(9,"TK_CHARCON wide character constant: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
L\'\\0[0-9]*\'          { 
                                pips_debug(9,"TK_CHARCON wide character constant: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }

\"([^\"]|(\"\")|(\\\"))*\"	{ 
                                pips_debug(9,"TK_STRINGCON regular or wide string: %s\n",yytext);
                                /* the "world" in L"Hello, " "world" should be treated as 
                                   wide even though there's no L immediately preceding it */
                                yylval.string = strdup(yytext);
				return(TK_STRINGCON);
                        }
L\"([^\"]|(\"\")|(\\\"))*\"	{ 
                                pips_debug(9,"TK_WSTRINGCON wide string: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_WSTRINGCON);
                        }
{floatnum}		{ 
                                pips_debug(9,"TK_FLOATCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_FLOATCON);
                        }
{hexnum}		{ 
                                pips_debug(9,"Hexnum TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{octnum}		{ 
                                pips_debug(9,"Octnum TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{intnum}		{ 
                                pips_debug(9,"TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
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
                                discard_C_comment();
				return(TK_LBRACE);
			}
"\}"			{
				pips_debug(9,"TK_RBRACE %s\n",yytext);
                                discard_C_comment();
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
{ident}			{ 
                                /* C keywords or identifiers */
				int t = is_c_keyword_typedef(yytext);
                                if (t>0) 
				   { 
				       pips_debug(9,"Keyword or typedef name: %s\n",yytext);
                                       yylval.string = strdup(yytext);
                                       /* Some of the pushes are unfortunately in cyacc.y */
                                       if(t==TK_IF || t==TK_SWITCH || t==TK_WHILE || t==TK_DO
                                          || t==TK_FOR)
                                         push_current_C_comment();
                                       /* Some comments are lost, especially in declarations */
                                       if(t==TK_ELSE || (t>=TK_CHAR && t <=TK_REGISTER))
                                         discard_C_comment();
				       return (t);
				   }
				else
				   {
				       pips_debug(9,"Identifier: %s\n",yytext);
                                       yylval.string = strdup(yytext);
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

void yyerror(char * msg)
{
    user_warning("yyerror", "C %s near \"%s\"\n", msg, yytext);
}

#ifndef yywrap
int yywrap() { return 1;}
#endif


