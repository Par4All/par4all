%{
#undef input()
#define input() (((yytchar=yysptr>yysbuf?U(*--yysptr):PipsGetc(yyin))==10?(yylineno++,yytchar):yytchar)==EOF?0:yytchar)

/* les definitions des valeurs des tokens */
#include <stdio.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"

#include "misc.h"

#include "syntax.h"

/* yacc generated header file
 */
#include "syn_yacc.h"

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

				pips_debug(9, "TK_KEYWORD %s %d %d\n", yytext, yyleng, tv);
				return(tv);
			}

{ULET}+({DIG}|{ULET})*	{ 
				pips_debug(9, "TK_NAME %s %d\n", 
					   yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_NAME); 
			}

{UICON}			{
				pips_debug(9, "TK_ICON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_ICON); 
			}

{RCON}			{
				pips_debug(9, "TK_RCON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_RCON); 
			}

{DCON}			{
				pips_debug(9, "TK_DCON %s %d\n",
					      yytext, yyleng);
                                yylval.string = strdup(yytext);
				return(TK_DCON); 
			}

\'([^\']|(\'\'))*\'	{
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
\n			{
				pips_debug(9, "TK_EOS\n");
				return(TK_EOS);
			}

.			{
   user_warning("sslex", 
                "[scanner] illegal character %c (hex %x)\n",
                yytext[0], yytext[0]);
   ParserError("Lexical Analysis", 
               "Check standard Fortran character set.\n");
			}
%%

void yyerror_lex_part(char *s)
{
    user_warning("parser"," error - %s near %s\n", s, yytext);
}
