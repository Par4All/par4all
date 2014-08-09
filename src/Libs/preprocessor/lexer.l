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

%option noinput
%option nounput
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
#include "splitc.h"
#include "preprocessor.h"


int csplit_line_number = 1; /**< To be exported to the parser for
				splitting the preprocessed file */
int user_line_number = 1; /**< To be exported to the parser for error
			      messages related to the user file */

/* To track current file position: */
#define YY_USER_ACTION update_csplit_file_offset_information();
size_t current_csplit_file_offset = 0;
size_t csplit_file_offset_beginning = 0;

/* A state-machine to gather the trailing comments of a statement with an
   heuristic to decide if some comments and spaces will go into the
   previous or next top-level construct: */
enum gather_comment_state_t {
  PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT, /**< Normal mode: comments and spaces
				     are associated to the top-level
				     construct to encounter: */
  GATHER_SPACES, /**< We gather only spaces in the previous top-level
		    construct */
  GATHER_COMMENT, /**< We gather comments in the previous top-level construct */
};
/* Indeed it does not work since csplit_copy() is called from splitc.y
   when a function definition is found and thus we cannot gather the
   comment in it... :-( */
enum gather_comment_state_t
  gather_comment_state_machine = PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT;


static void gdb_marker()
{
/* Just to be able to set an easy breakpoint */
;
}


/* Reinitialise global position numbers for a new file. */
void reset_csplit_line_number()
{
  csplit_line_number = 1;
  user_line_number = 1;
  current_csplit_file_offset = 0;
  csplit_file_offset_beginning = 0;
  gather_comment_state_machine = PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT;
}

static int csplit_current_beginning = 1000000;
static int user_current_beginning = 1000000;


/* If we encounter a possible function begin, just snapshot the position
   of the function beginning: */
static void update_csplit_current_beginning() {
  user_current_beginning = csplit_current_beginning > csplit_line_number?
    user_line_number : user_current_beginning;
  csplit_current_beginning = csplit_current_beginning > csplit_line_number?
    csplit_line_number : csplit_current_beginning;
  /* From now, the comments and spaces are associated to the current
     top-level construct: */
  gather_comment_state_machine = PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT;
}


/* This function is called at the end of any top-level C construct: */
void reset_csplit_current_beginning()
{
  int leaked = 0;

  csplit_current_beginning = 1000000;
  user_current_beginning = 1000000;
  csplit_is_static_p = false;

  /* Snapshot the current file position as a potential function begin: */
  csplit_file_offset_beginning = current_csplit_file_offset;

  /* Aggregate following spaces and comments on the sameline to the
     previous top-level construct: */
  gather_comment_state_machine = GATHER_COMMENT;


  if(!string_undefined_p(csplit_current_function_name))
    free(csplit_current_function_name);
  csplit_current_function_name = string_undefined;

  if(!string_undefined_p(csplit_current_function_name2))
    free(csplit_current_function_name2);
  csplit_current_function_name2 = string_undefined;

  if(!string_undefined_p(csplit_definite_function_name))
    free(csplit_definite_function_name);
  csplit_definite_function_name = string_undefined;

  if(!string_undefined_p(csplit_definite_function_signature))
    free(csplit_definite_function_signature);
  csplit_definite_function_signature = string_undefined;

  if((leaked=check_signature_balance())!=0) {
    /* FI: I'm not ready to abort() here... */
    pips_debug(5, "Likely memory leaks: %d\n", leaked);
  }
}

int get_csplit_current_beginning()
{
  return csplit_current_beginning;
}


/* Get the current line number in the file to split from the user point of
   view (before preprocessor expansion): */
int get_user_current_beginning() {
  return user_current_beginning;
}


/* Get the file position in the file to split where the current function
   begins: */
size_t get_csplit_file_offset_beginning() {
  return csplit_file_offset_beginning;
}


/* Get the current file position in the file to split: */
size_t get_current_csplit_file_offset() {
  return current_csplit_file_offset;
}


/* Function called each time a token is read to trac file position
   information: */
void update_csplit_file_offset_information() {
  current_csplit_file_offset += yyleng;
}


static int lf_count(string s)
{
  string cs = s;
  int count = 0;

  while(*cs) {
    if(*cs=='\n') count++;
    cs++;
  }
  return count;
}

%}

/* To track file line number automatically: */
%option yylineno

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

("/*"([^*]|("*"+[^/*]))*"*"+"/")|("//"("\\\n"|[^\n])*"\n"?)	{
	 // Bug in the Flex 2.5.35 manual A.4.3, "/***/" fails. :-(
			  csplit_line_number += lf_count(yytext);
			  user_line_number += lf_count(yytext);


			  pips_debug(5,"Comment \"%s\"\n",yytext);

			  if (gather_comment_state_machine == GATHER_COMMENT)
			    /* This comment is associated to previous
			       top-level construct: */
			    reset_csplit_current_beginning();
			  if (gather_comment_state_machine == GATHER_SPACES)
			    /* We encounter a comment whereas we were
			       gathering spaces into the previous
			       top-level construct. So now this comment
			       will go in the next top-level construct: */
			    gather_comment_state_machine = PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT;
                        }

^"#"("\\\n"|[^\n])*              {
                /* Look for a # up to the end of line,
                   dealing with any backslashed new lines: */
                                /* csplit_line_number++; */
                                pips_debug(5,"Pragma comment %s\n",yytext);
                                /* Must be a line pragma left by the preprocessor:
                                   # 1 blablabla.c
                                   #pragma blablabla
                                   #line nnn
                                */
                                gdb_marker();
                                size_t n;
                                if(strstr(yytext, "#pragma")==yytext) {
                                   user_line_number--;
                                }
                                /* the preprocessor seems to provide line information in pragmas
                                   which are not #line pragmas. */
                                else if(/* strstr(yytext, "#line")==yytext && */ (n=strlen(yytext))>=4) {
                                   int initial_C_line_number = -1;
								   char include_file[n] ;
                                   int items = sscanf(yytext+1, "%d \"%[^\"]\"", &initial_C_line_number,include_file);
                                   if(items>=1) {
                                      /* Get rid of the pragma LF itslef */
                                      user_line_number = initial_C_line_number-1;
                                      if(items==2) {
                                        if(!current_file_path) current_file_path = strdup(include_file);
                                        else if(!current_include_file_path) current_include_file_path=strdup(include_file);
                                        else if(strcmp(include_file,current_file_path)==0) { free(current_include_file_path) ; current_include_file_path=NULL; }
                                      }
                                   }
                                   else {
                                      pips_user_warning("No line number in # pragma: \"%s\".\n", yytext);
                                      CParserError("Ill. formated # pragma\n");
                                   }
                                }
                        }

\n                      {
                          csplit_line_number++;
			  user_line_number++;
			  pips_debug(5, "New line %d, ext=%d, func=%d\n", csplit_line_number, csplit_is_external, csplit_is_function);

			  if (gather_comment_state_machine == GATHER_COMMENT) {
			    /* We encounter a newline, stop gathering
			       comments and gather spaces from now: */
			    //gather_comment_state_machine = GATHER_SPACES;
			    /* Well, ideed, we want to be compatible with
			       old behaviour: associate newlines and other
			       from now to next top-level construct: */
			    gather_comment_state_machine = PUT_IN_NEXT_TOP_LEVEL_CONSTRUCT;
			    /* Associate the current \n to the previous
			       top-level construct: */
			    reset_csplit_current_beginning();
			  }
			  if (gather_comment_state_machine == GATHER_SPACES) {
			    /* Add this newline to the previous top-level
			       construct: */
			     reset_csplit_current_beginning();
			  }
                        }

[[:blank:]]		{
                          /* Eat up whitespaces. After the previous '\n'
			     since it includes '\n' and we want
			     lower priority */
			  if (gather_comment_state_machine == GATHER_COMMENT
			      || gather_comment_state_machine == GATHER_SPACES) {
			    /* Add this space to the previous top-level
			       construct: */
			     reset_csplit_current_beginning();
			  }
                        }

"_Pragma" 	        {
                                return(TK_PRAGMA);
                        }

\'([^\\]|{char_escape}|{oct_escape}|{hex_escape}|{universal-character-name})\'           {
  /* Have a look to Flex A.4.3 too... */
                                /* Escaped character constants. Their
                                   syntax is understood in PIPS character
                                   constant construtors. */
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }

L\'([^\\]|{char_escape}|{oct_escape}|{wide_hex_escape}|{universal-character-name})\'           {
                                /* Escaped wide character constants. Their
                                   syntax is understood in PIPS character
                                   constant construtors. */
                                pips_debug(9,"TK_CHARCON wide character constant: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }

\"(\\\"|[^\"\\]|\\[^\"])*\"	{
                                pips_debug(5,"TK_STRINGCON regular or wide string: %s\n",yytext);
                                /* the "world" in L"Hello, " "world" should be treated as
                                   wide even though there's no L immediately preceding it */
                                yylval.string = strdup(yytext);
				return(TK_STRINGCON);
                        }

L\"(\\\"|[^\"\\]|\\[^\"])*\"	{
                                pips_debug(5,"TK_WSTRINGCON wide string: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_WSTRINGCON);
                        }
{floatnum}		{
                                pips_debug(5,"TK_FLOATCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_FLOATCON);
                        }
{complexnum}		{
                                pips_debug(5,"TK_COMPLEXCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_COMPLEXCON);
                        }
{hexnum}		{
                                pips_debug(5,"Hexnum TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{octnum}		{
                                pips_debug(5,"Octnum TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_INTCON);
                        }
{intnum}		{
                                pips_debug(5,"TK_INTCON %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_INTCON);
                        }
"!quit!"		{
				pips_debug(5,"TK_EOF %s\n",yytext);
				yyterminate();
			}
"..."			{
				pips_debug(5,"TK_ELLIPSIS %s\n",yytext);
				return(TK_ELLIPSIS);
			}
"+="			{
				pips_debug(5,"TK_PLUS_EQ %s\n",yytext);
				return(TK_PLUS_EQ);
			}
"-="			{
				pips_debug(5,"TK_MINUS_EQ %s\n",yytext);
				return(TK_MINUS_EQ);
			}
"*="			{
				pips_debug(5,"TK_STAR_EQ %s\n",yytext);
				return(TK_STAR_EQ);
			}
"/="			{
				pips_debug(5,"TK_SLASH_EQ %s\n",yytext);
				return(TK_SLASH_EQ);
			}
"%="			{
				pips_debug(5,"TK_PERCENT_EQ %s\n",yytext);
				return(TK_PERCENT_EQ);
			}
"|="			{
				pips_debug(5,"TK_PIPE_EQ %s\n",yytext);
				return(TK_PIPE_EQ);
			}
"&="			{
				pips_debug(5,"TK_AND_EQ %s\n",yytext);
				return(TK_AND_EQ);
			}
"^="			{
				pips_debug(5,"TK_CIRC_EQ %s\n",yytext);
				return(TK_CIRC_EQ);
			}
"<<="			{
				pips_debug(5,"TK_INF_INF_EQ %s\n",yytext);
				return(TK_INF_INF_EQ);
			}
">>="			{
				pips_debug(5,"TK_SUP_SUP_EQ %s\n",yytext);
				return(TK_SUP_SUP_EQ);
			}
"<<"			{
				pips_debug(5,"TK_INF_INF %s\n",yytext);
				return(TK_INF_INF);
			}
">>"			{
				pips_debug(5,"TK_SUP_SUP %s\n",yytext);
				return(TK_SUP_SUP);
			}
"=="			{
				pips_debug(5,"TK_EQ_EQ %s\n",yytext);
				return(TK_EQ_EQ);
			}
"!="			{
				pips_debug(5,"TK_EXCLAM_EQ %s\n",yytext);
				return(TK_EXCLAM_EQ);
			}
"<="			{
				pips_debug(5,"TK_INF_EQ %s\n",yytext);
				return(TK_INF_EQ);
			}
">="			{
				pips_debug(5,"TK_SUP_EQ %s\n",yytext);
				return(TK_SUP_EQ);
			}
"="			{
				pips_debug(5,"TK_EQ %s\n",yytext);
				return(TK_EQ);
			}
"<"			{
				pips_debug(5,"TK_INF %s\n",yytext);
				return(TK_INF);
			}
">"			{
				pips_debug(5,"TK_SUP %s\n",yytext);
				return(TK_SUP);
			}
"++"			{
				pips_debug(5,"TK_PLUS_PLUS %s\n",yytext);
				return(TK_PLUS_PLUS);
			}
"--"			{
				pips_debug(5,"TK_MINUS_MINUS %s\n",yytext);
				return(TK_MINUS_MINUS);
			}
"->"			{
				pips_debug(5,"TK_ARROW %s\n",yytext);
				return(TK_ARROW);
			}
"+"			{
				pips_debug(5,"TK_PLUS %s\n",yytext);
				return(TK_PLUS);
			}
"-"			{
				pips_debug(5,"TK_MINUS %s\n",yytext);
				return(TK_MINUS);
			}
"*"			{
				pips_debug(5,"TK_STAR %s\n",yytext);
				return(TK_STAR);
			}
"/"			{
				pips_debug(5,"TK_SLASH %s\n",yytext);
				return(TK_SLASH);
			}
"%"			{
				pips_debug(5,"TK_PERCENT %s\n",yytext);
				return(TK_PERCENT);
			}
"!"			{
				pips_debug(5,"TK_EXCLAM %s\n",yytext);
				return(TK_EXCLAM);
			}
"&&"			{
				pips_debug(5,"TK_AND_AND %s\n",yytext);
				return(TK_AND_AND);
			}
"||"			{
				pips_debug(5,"TK_PIPE_PIPE %s\n",yytext);
				return(TK_PIPE_PIPE);
			}
"&"			{
				pips_debug(5,"TK_AND %s\n",yytext);
				return(TK_AND);
			}
"|"			{
				pips_debug(5,"TK_PIPE %s\n",yytext);
				return(TK_PIPE);
			}
"^"			{
				pips_debug(5,"TK_CIRC %s\n",yytext);
				return(TK_CIRC);
			}
"?"			{
				pips_debug(5,"TK_QUEST %s\n",yytext);
				return(TK_QUEST);
			}
":"			{
				pips_debug(5,"TK_COLON %s\n",yytext);
				return(TK_COLON);
			}
"~"			{
				pips_debug(5,"TK_TILDE %s\n",yytext);
				return(TK_TILDE);
			}
"\{"                    {
				pips_debug(5,"TK_LBRACE %s\n",yytext);
				return(TK_LBRACE);
			}
"\}"			{
				pips_debug(5,"TK_RBRACE %s\n",yytext);
				return(TK_RBRACE);
			}
"["			{
				pips_debug(5,"TK_LBRACKET %s\n",yytext);
				return(TK_LBRACKET);
			}
"]"			{
				pips_debug(5,"TK_RBRACKET %s\n",yytext);
				return(TK_RBRACKET);
			}
"("			{
				pips_debug(5,"TK_LPAREN %s\n",yytext);
				return(TK_LPAREN);
			}
")"			{
				pips_debug(5,"TK_RPAREN %s\n",yytext);
				return(TK_RPAREN);
			}
";"			{
				pips_debug(5,"TK_SEMICOLON %s\n",yytext);
				return(TK_SEMICOLON);
			}
","			{
				pips_debug(5,"TK_COMMA %s\n",yytext);
				return(TK_COMMA);
			}
"."			{
				pips_debug(5,"TK_DOT %s\n",yytext);
				return(TK_DOT);
			}
"sizeof"		{
				pips_debug(5,"TK_SIZEOF %s\n",yytext);
				return(TK_SIZEOF);
			}
"__asm"                 {
				pips_debug(5,"TK_ASM %s\n",yytext);
				return(TK_ASM);
			}
"__va_list" 	        {
	/* convert __va_list from bsd into __builtin_va_list */
	pips_debug(5,"TK_VA_LIST %s\n",yytext);
	yytext = strdup("__builtin_va_list");
	int t = is_c_keyword_typedef(yytext);
	update_csplit_current_beginning();
	pips_assert("builtin valist found\n",t>0);
	pips_debug(5,"Keyword or typedef name: %s\n",yytext);
	yylval.string = strdup(yytext);
	return (t);
}
"__attribute__"                 {

/* FI: when we change our minds and want to make these tokens, we must
 * insert them in keyword_typedef_table.
 */

				pips_debug(5,"TK_ATTRIBUTE %s\n",yytext);
				return(TK_ATTRIBUTE);
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
                                /* C keywords or identifiers */
				int t = is_c_keyword_typedef(yytext);
                                update_csplit_current_beginning();
                                if (t>0)
				   {
				       pips_debug(5,"Keyword or typedef name: %s\n",yytext);
                                       yylval.string = strdup(yytext);
				       return (t);
				   }
				else
				   {
 				       pips_debug(5,"TK_IDENT: %s\n",yytext);
                                       /* Might not work if a function returns a struct declared locally in the function declaration, or a pointer to such a structure */
                                       if(string_undefined_p(csplit_current_function_name)) {
					 csplit_current_function_name = strdup(yytext);
					 pips_debug(5, "Temptative function name found: \"%s\"\n",
						    csplit_current_function_name);
                                       }
                                       else if(string_undefined_p(csplit_current_function_name2)) {
				         pips_debug(5, "Identifier \"%s\" ignored because of two previous identifiers \"%s\"\n",
						     yytext, csplit_current_function_name);
				         csplit_current_function_name2 = strdup(yytext);
				       }
				       else {
				         free(csplit_current_function_name2);
					 pips_debug(5, "Identifier \"%s\" not ignored in spite of two previous identifiers \"%s\"\n",
						    yytext, csplit_current_function_name);
					 csplit_current_function_name2 = strdup(yytext);
				       }

				       yylval.string = strdup(yytext);
				       return(TK_IDENT);
				   }
			}
<<EOF>>			{
				pips_debug(5,"TK_EOF %s\n",yytext);
				return(TK_EOF);
			}
.			{
                                if(*yytext=='#') {
			          pips_user_warning("Unrecognized character %s\n",yytext);
			          pips_user_warning("Preprocessor directives must appear first in a line.\n"
"For instance, insert a LF if a comment precedes a directive.");
                                }
				else
			          pips_user_warning("Unrecognized character %s\n",yytext);
			}
%%

/* This function is renamed splitc_error(). It should probably reset more variables */
void yyerror(const char * msg)
{
    user_warning("yyerror",
                 "C %s near \"%s\" at preprocessed line %d (user line %d)\n",
                 msg, yytext, csplit_line_number, user_line_number);
    /* I should call csplit_parser_error() or the equivalent module */
    pips_user_error("Syntax error\n");
}

#ifndef yywrap
int yywrap() { return 1;}
#endif
