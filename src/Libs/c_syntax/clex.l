/* $Id$
   $Log: clex.l,v $
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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "c_syntax.h"
#include "cyacc.h"

struct Ckeyword {
	char * keystr;
	int keyval;
};

#define KEYNUM 34

struct Ckeyword keytable[KEYNUM] = {
{"auto", TK_AUTO},
{"break", TK_BREAK},
{"case", TK_CASE},
{"char", TK_CHAR},
{"const", TK_CONST},
{"continue", TK_CONTINUE},
{"default", TK_DEFAULT},
{"do", TK_DO},
{"double", TK_DOUBLE},
{"else", TK_ELSE},
{"enum", TK_ENUM},
{"extern", TK_EXTERN},
{"float", TK_FLOAT},
{"for", TK_FOR},
{"goto", TK_GOTO},
{"if", TK_IF},
{"inline", TK_INLINE},
{"int", TK_INT},
{"long", TK_LONG},
{"register", TK_REGISTER},
{"restrict", TK_RESTRICT},
{"return", TK_RETURN},
{"short", TK_SHORT},
{"signed", TK_SIGNED},
{"sizeof", TK_SIZEOF},
{"static", TK_STATIC},
{"struct", TK_STRUCT},
{"switch", TK_SWITCH},
{"typedef", TK_TYPEDEF},
{"union", TK_UNION},
{"unsigned", TK_UNSIGNED},
{"void", TK_VOID},
{"volatile", TK_VOLATILE},
{"while", TK_WHILE}};
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
exponent ([eE]["+""-"]?{decdigit}+)
fraction ("."{decdigit}+)
decfloat (({intnum}?{fraction})|({intnum}{exponent})|({intnum}?{fraction}{exponent})|({intnum}'.')|({intnum}'.'{exponent})) 
hexfraction (({hexdigit}*"."{hexdigit}+)|({hexdigit}+))
binexponent ([pP]["+""-"]?{decdigit}+)
hexfloat (({hexprefix}{hexfraction}{binexponent})|({hexprefix}{hexdigit}+{binexponent}))
floatsuffix [fFlL]
floatnum (({decfloat}|{hexfloat}){floatsuffix}?)
ident ({letter}|"_")({letter}|{decdigit}|"_")* 
attribident (({letter}|"_")({letter}|{decdigit}|"_"|":"))
blank [ \t\012\r]+
escape ('\\'_)
hex_escape ('\\'[xX]{hexdigit}+)
oct_escape ('\\'{octdigit}{octdigit}?{octdigit}?) 

%%
"/*"([^\*]|(\*+[^\/]))*"*/"	{ 
                                printf("Comment %s\n",yytext);
                        }
"//".*\n                { 
                                printf("Comment %s\n",yytext);
                        }
{blank}			{ 
                                /* Eat up whitespaces */
                        }
'\n'                    { 
                                printf("New line\n");
                        }
"#"			{ 
                                printf("Hash %s\n",yytext);
                        }
"_Pragma" 	        { 
                                printf("Pragma not implemented %s\n",yytext);
				return(TK_PRAGMA);
                        }
\'([^\']|(\'\')|(\\\'))*\'	{ 
                                /* Attention, the regular expression is not true*/
                                printf("TK_CHARCON: %s\n",yytext);
				return(TK_CHARCON);
                        }
L\'([^\']|(\'\')|(\\\'))*\'	{ 
                                /* Attention, the regular expression is not true*/
                                printf("TK_CHARCON wide character constant: %s\n",yytext);
				return(TK_CHARCON);
                        }
\"([^\"]|(\"\")|(\\\"))*\"	{ 
                                printf("TK_STRINGCON regular or wide string: %s\n",yytext);
                                /* the "world" in L"Hello, " "world" should be treated as 
                                   wide even though there's no L immediately preceding it */
				return(TK_STRINGCON);
                        }
L\"([^\"]|(\"\")|(\\\"))*\"	{ 
                                printf("TK_WSTRINGCON wide string: %s\n",yytext);
				return(TK_WSTRINGCON);
                        }
{floatnum}		{ 
                                printf("TK_FLOATCON %s\n",yytext);
				return(TK_FLOATCON);
                        }
{hexnum}		{ 
                                printf("Hexnum TK_INTCON %s\n",yytext);
				return(TK_INTCON);
                        }
{octnum}		{ 
                                printf("Octnum TK_INTCON %s\n",yytext);
				return(TK_INTCON);
                        }
{intnum}		{ 
                                printf("TK_INTCON %s\n",yytext);
				return(TK_INTCON);
                        }
"!quit!"		{
				printf("EOF %s\n",yytext);
				return(TK_EOF);
			}
"..."			{
				printf("TK_ELLIPSIS %s\n",yytext);
				return(TK_ELLIPSIS);
			}
"+="			{
				printf("TK_PLUS_EQ %s\n",yytext);
				return(TK_PLUS_EQ);
			}
"-="			{
				printf("TK_MINUS_EQ %s\n",yytext);
				return(TK_MINUS_EQ);
			}
"*="			{
				printf("TK_STAR_EQ %s\n",yytext);
				return(TK_STAR_EQ);
			}
"/="			{
				printf("TK_SLASH_EQ %s\n",yytext);
				return(TK_SLASH_EQ);
			}
"%="			{
				printf("TK_PERCENT_EQ %s\n",yytext);
				return(TK_PERCENT_EQ);
			}
"|="			{
				printf("TK_PIPE_EQ %s\n",yytext);
				return(TK_PIPE_EQ);
			}
"&="			{
				printf("TK_AND_EQ %s\n",yytext);
				return(TK_AND_EQ);
			}
"^="			{
				printf("TK_CIRC_EQ %s\n",yytext);
				return(TK_CIRC_EQ);
			}
"<<="			{
				printf("TK_INF_INF_EQ %s\n",yytext);
				return(TK_INF_INF_EQ);
			}
">>="			{
				printf("TK_SUP_SUP_EQ %s\n",yytext);
				return(TK_SUP_SUP_EQ);
			}
"<<"			{
				printf("TK_INF_INF %s\n",yytext);
				return(TK_INF_INF);
			}
">>"			{
				printf("TK_SUP_SUP %s\n",yytext);
				return(TK_SUP_SUP);
			}
"=="			{
				printf("TK_EQ_EQ %s\n",yytext);
				return(TK_EQ_EQ);
			}
"!="			{
				printf("TK_EXCLAM_EQ %s\n",yytext);
				return(TK_EXCLAM_EQ);
			}
"<="			{
				printf("TK_INF_EQ %s\n",yytext);
				return(TK_INF_EQ);
			}
">="			{
				printf("TK_SUP_EQ %s\n",yytext);
				return(TK_SUP_EQ);
			}
"="			{
				printf("TK_EQ %s\n",yytext);
				return(TK_EQ);
			}
"<"			{
				printf("TK_INF %s\n",yytext);
				return(TK_INF);
			}
">"			{
				printf("TK_SUP %s\n",yytext);
				return(TK_SUP);
			}
"++"			{
				printf("TK_PLUS_PLUS %s\n",yytext);
				return(TK_PLUS_PLUS);
			}
"--"			{
				printf("TK_MINUS_MINUS %s\n",yytext);
				return(TK_MINUS_MINUS);
			}
"->"			{
				printf("TK_ARROW %s\n",yytext);
				return(TK_ARROW);
			}
"+"			{
				printf("TK_PLUS %s\n",yytext);
				return(TK_PLUS);
			}
"-"			{
				printf("TK_MINUS %s\n",yytext);
				return(TK_MINUS);
			}
"*"			{
				printf("TK_STAR %s\n",yytext);
				return(TK_STAR);
			}
"/"			{
				printf("TK_SLASH %s\n",yytext);
				return(TK_SLASH);
			}
"%"			{
				printf("TK_PERCENT %s\n",yytext);
				return(TK_PERCENT);
			}
"!"			{
				printf("TK_EXCLAM %s\n",yytext);
				return(TK_EXCLAM);
			}
"&&"			{
				printf("TK_AND_AND %s\n",yytext);
				return(TK_AND_AND);
			}
"||"			{
				printf("TK_PIPE_PIPE %s\n",yytext);
				return(TK_PIPE_PIPE);
			}
"&"			{
				printf("TK_AND %s\n",yytext);
				return(TK_AND);
			}
"|"			{
				printf("TK_PIPE %s\n",yytext);
				return(TK_PIPE);
			}
"^"			{
				printf("TK_CIRC %s\n",yytext);
				return(TK_CIRC);
			}
"?"			{
				printf("TK_QUEST %s\n",yytext);
				return(TK_QUEST);
			}
":"			{
				printf("TK_COLON %s\n",yytext);
				return(TK_COLON);
			}
"~"			{
				printf("TK_TILDE %s\n",yytext);
				return(TK_TILDE);
			}	
"\{"                    {
				printf("TK_LBRACE %s\n",yytext);
				return(TK_LBRACE);
			}
"\}"			{
				printf("TK_RBRACE %s\n",yytext);
				return(TK_RBRACE);
			}
"["			{
				printf("TK_LBRACKET %s\n",yytext);
				return(TK_LBRACKET);
			}
"]"			{
				printf("TK_RBRACKET %s\n",yytext);
				return(TK_RBRACKET);
			}
"("			{
				printf("TK_LPAREN %s\n",yytext);
				return(TK_LPAREN);
			}
")"			{
				printf("TK_RPAREN %s\n",yytext);
				return(TK_RPAREN);
			}
";"			{
				printf("TK_SEMICOLON %s\n",yytext);
				return(TK_SEMICOLON);
			}
","			{
				printf("TK_COMMA %s\n",yytext);
				return(TK_COMMA);
			}
"."			{
				printf("TK_DOT %s\n",yytext);
				return(TK_DOT);
			}
"sizeof"		{
				printf("TK_SIZEOF %s\n",yytext);
				return(TK_SIZEOF);
			}
"__asm"                 {
				printf("TK_ASM %s\n",yytext);
				return(TK_ASM);
			} 
{ident}			{ 
                                /* C keywords or identifiers */
				int t = is_c_keyword(yytext);
                                if (t>0) 
				   { 
				       printf("Keyword %s\n",yytext);
				       return (t);
				   }
				else
				   {
				       printf("Identifier %s\n",yytext);
				       return(TK_IDENT); 
				   }
			}
<<EOF>>			{
				printf("TK_EOF %s\n",yytext);
				return(TK_EOF);
			}
.			{
			        printf("Unrecognized character %s\n",yytext);
			}
%%

/* This function checks if s is a C keyword or not. 
   It returns an integer number corresponding to the keyword. 
   It returns 0 if s is not a keyword */
int is_c_keyword(char * s)
{
    register int i, c;
    char *t;
    char buffer[32]; /* The length limit is 32*/

    /* The string s is changed to lowercases */
    t = buffer;
    while ((c = *s++) ) {
      if (isupper(c))
	    c = tolower(c);
	*t++ = c;
    }
    *t = '\0';

    for (i=0; i<KEYNUM; i++)
        {
	  if (strcmp(buffer,keytable[i].keystr) == 0) 
	      return(keytable[i].keyval);
	}
    return 0;
}


void yyerror(char * msg)
{
    int c;

    /* Do not use pips_user_warning() to avoid portability issues */
    user_warning("syn_error", "Syntax error - %s near \"%s\"\n", msg, yytext);

    /* what does it mean? */
    user_warning("syn_error", "Non analyzed source text stored in logfile\n");
    
    while ((c = getc(yyin)) != EOF)
	putc(c, stderr);
	
    /* pas de recouvrement d'erreurs */
    ParserError("yyerror", "Syntax error\n");
}

#ifndef yywrap
yywrap() { return 0;}
#endif


