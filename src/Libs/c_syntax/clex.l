/* $Id$
   $Log: clex.l,v $
   Revision 1.5  2003/09/05 14:19:52  nguyen
   Modify pips debug levels

   Revision 1.4  2003/08/13 08:00:33  nguyen
   Correct the regular expressions for floating number, ...

   Revision 1.3  2003/08/06 14:12:44  nguyen
   Correct pragma comment

   Revision 1.2  2003/08/04 14:20:59  nguyen
   Preliminary version of the C parser

   Revision 1.1  2003/06/24 08:48:47  nguyen
   Initial revision
 */

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

/* To avoid warnings */
extern char *strdup(const char *s1);
extern int fileno(FILE *stream);
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
                        }
"//"[^\n]*              { 
                                pips_debug(9,"Comment %s\n",yytext);
                        }
^"#"[^\n]*              { 
                                pips_debug(9,"Pragma comment %s\n",yytext);
                        }
{blank}			{ 
                                /* Eat up whitespaces */
                        }
\n                      { 
                                pips_debug(9,"New line\n");
                        }

"_Pragma" 	        { 
                                pips_debug(9,"Pragma not implemented %s\n",yytext);
				return(TK_PRAGMA);
                        }
\'([^\']|(\'\')|(\\\'))*\'	{ 
                                /* Attention, the regular expression is not true*/
                                pips_debug(9,"TK_CHARCON: %s\n",yytext);
                                yylval.string = strdup(yytext);
				return(TK_CHARCON);
                        }
L\'([^\']|(\'\')|(\\\'))*\'	{ 
                                /* Attention, the regular expression is not true*/
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
				return(TK_LBRACE);
			}
"\}"			{
				pips_debug(9,"TK_RBRACE %s\n",yytext);
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
				       return (t);
				   }
				else
				   {
				       pips_debug(9,"Identifier: %s\n",yytext);
                                       yylval.string = strdup(yytext);
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


