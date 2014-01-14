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

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

/* UICON and ICON must be distinguished from RCON and DCON:
      REAL*8 D1
      REAL*8 E2
   Should not be parsed as real constants!
*/

#ifdef FLEX_SCANNER

/* flex incompatible way of playing with input characters:-)
 */

#define YY_INPUT(buf,result,max_size) \
{ int c = PipsGetc(yyin); result = (c == EOF)? YY_NULL: (buf[0] = c, 1);}

#else /* we're not using flex... let us assume some very peculiar lex... */

/* the portability of the following macro makes many (:-) assumptions
 * about lex internals hence should not be very portable...
 * POSIX states that input cannot be redefined.
 * the only possible posix implementation would be to refilter the
 * file through PipsGetc and then to use a standard lex/yacc.
 */
#undef input()
#define input() (((yytchar=yysptr>yysbuf?U(*--yysptr):PipsGetc(yyin))==10?(yylineno++,yytchar):yytchar)==EOF?0:yytchar)

#endif /* FLEX_SCANNER */

#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "syntax.h"

/* yacc generated header file
 */
#include "syn_yacc.h"

/* stdlib defines abs on solaris 2 */
#ifdef abs
#undef abs
#endif

static int line = 1; /* position in pseudo-file... */

%}

DIG [0-9]
ULET [A-Z_]
LLET [a-z_]
SIG [+-]
UICON ({DIG}+)
ICON ({SIG}?{UICON})
MANT (({DIG}+\.{DIG}*)|({DIG}*\.{DIG}+))
REXP (E{ICON})
DEXP (D{ICON})
RCON (({MANT}|{UICON}){REXP}?)
DCON (({MANT}|{UICON}){DEXP}?)

%start TYPEDECLARATION

%%
%AND%			{
				pips_debug(9, "TK_AND\n");
				return(TK_AND);
			}
%INOUT%			{
				pips_debug(9, "TK_INOUT\n");
				return(TK_INOUT);
			}
%IN%			{
				pips_debug(9, "TK_IN\n");
				return(TK_IN);
			}
%OUT%			{
				pips_debug(9, "TK_OUT\n");
				return(TK_OUT);
			}
%EQ%			{
				pips_debug(9, "TK_EQ\n");
				return(TK_EQ);
			}
%EQV%			{
				pips_debug(9, "TK_EQV\n");
				return(TK_EQV);
			}
%GE%			{
				pips_debug(9, "TK_GE\n");
				return(TK_GE);
			}
%GT%			{
				pips_debug(9, "TK_GT\n");
				return(TK_GT);
			}
%LE%			{
				pips_debug(9, "TK_LE\n");
				return(TK_LE);
			}
%LT%			{
				pips_debug(9, "TK_LT\n");
				return(TK_LT);
			}
%NE%			{
				pips_debug(9, "TK_NE\n");
				return(TK_NE);
			}
%NEQV%			{
				pips_debug(9, "TK_NEQV\n");
				return(TK_NEQV);
			}
%NOT%			{
				pips_debug(9, "TK_NOT\n");
				return(TK_NOT);
			}
%OR%			{
				pips_debug(9, "TK_OR\n");
				return(TK_OR);
			}
%TRUE%			{
				pips_debug(9, "TK_TRUE\n");
				return(TK_TRUE);
			}
%FALSE%			{
				pips_debug(9, "TK_FALSE\n");
				return(TK_FALSE);
			}

{ULET}{LLET}+		{
				int tv = IsCapKeyword(yytext);
				if (tv == TK_FORMAT) {
                                        char * s = FormatValue;
					while ((*s++ = input()) != '\n') ;
					unput('\n');
                                        *(s-1) = '\0';
				}

                                switch(tv) {
                                case TK_CHARACTER:
                                case TK_COMPLEX:
                                case TK_INTEGER:
                                case TK_LOGICAL:
                                case TK_REAL:
                                        BEGIN TYPEDECLARATION;
                                }

				pips_debug(9, "TK_KEYWORD %s %d %d\n", yytext, yyleng, tv);
				return(tv);
			}

{ULET}+({DIG}|{ULET})*	{ 
				pips_debug(9, "TK_NAME %s %d\n", 
					   yytext, yyleng);
                                yylval.string = strdup(yytext);
                                BEGIN INITIAL;
				return(TK_NAME); 
			}

{UICON}			{
				pips_debug(9, "TK_ICON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_ICON); 
			}

<INITIAL>{RCON}			{
				pips_debug(9, "TK_RCON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_RCON); 
			}

<INITIAL>{DCON}		{
				pips_debug(9, "TK_DCON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_DCON); 
			}

\'([^\']|(\'\')|(\\\'))*\'	{
				pips_debug(9, "TK_SCON %s %d\n",
                                           yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_SCON);
			}
\"([^\"]|(\"\")|(\\\"))*\"	{
				pips_debug(9, "TK_SCON %s %d\n",
                                           yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_SCON);
			}
"-"			{
				pips_debug(9, "TK_MINUS\n");
				return(TK_MINUS);
			}
"+"			{
				pips_debug(9, "TK_PLUS\n");
				return(TK_PLUS);
			}
"/"			{
				pips_debug(9, "TK_SLASH\n");
				return(TK_SLASH);
			}
"**"			{
				pips_debug(9, "TK_POWER\n");
				return(TK_POWER);
			}
"*"			{
				pips_debug(9, "TK_STAR\n");
				return(TK_STAR);
			}
"("			{
				pips_debug(9, "TK_LPAR\n");
				return(TK_LPAR);
			}
")"			{
				pips_debug(9, "TK_RPAR\n");
				return(TK_RPAR);
			}
","			{
				pips_debug(9, "TK_COMMA\n");
				return(TK_COMMA);
			}
":"			{
				pips_debug(9, "TK_COLON\n");
				return(TK_COLON);
			}
"="			{
				pips_debug(9, "TK_EQUALS\n");
				return(TK_EQUALS);
			}
"$"			{
				pips_debug(9, "TK_CURRENCY\n");
				return(TK_CURRENCY);
			}
"//"			{
				pips_debug(9, "TK_CONCAT\n");
				return(TK_CONCAT);
			}
\n			{       line++;
                                BEGIN INITIAL;
				pips_debug(9, "TK_EOS\n");
				return(TK_EOS);
			}
.			{
                                /* Ignore carriage return from VMS and Microsoft and Apple?
                                 * Not here! It must have been done earlier in the processing.
                                 */
                                pips_user_warning("illegal character \"%c\" (hexadecimal %#04x)\n",
                                                  yytext[0], yytext[0]);
                                ParserError("Lexical Analysis", 
                                "Check standard Fortran character set.\n");
			}
%%

void yyerror(const char * msg)
{
    int c;

    /* Do not use pips_user_warning() to avoid portability issues */

    user_warning("syn_error", "Syntax error - %s line %d near \"%s\"\n", 
		      msg, line, yytext);

    /* what does it mean? */
    user_warning("syn_error", "Non analyzed source text stored in logfile\n");
    
    while ((c = getc(yyin)) != EOF)
	putc(c, stderr);
	
    /* pas de recouvrement d'erreurs */
    ParserError("yyerror", "Syntax error\n");
}

void yyreset_lex(void)
{
    line = 0;
#ifdef FLEX_SCANNER
    yyrestart(yyin);
#else
    /* reset lex... Might be better to read the whole file?
     */
    syn_sptr = syn_sbuf;
# define MMNEWLINE 10
    syn_previous = MMNEWLINE;
#endif
}

/* end of it */
