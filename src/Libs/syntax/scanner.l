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
#include "toklex.h"
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
				debug(9, "", "TK_AND\n");
				return(TK_AND);
			}
%INOUT%			{
				debug(9, "", "TK_INOUT\n");
				return(TK_INOUT);
			}
%IN%			{
				debug(9, "", "TK_IN\n");
				return(TK_IN);
			}
%OUT%			{
				debug(9, "", "TK_OUT\n");
				return(TK_OUT);
			}
%EQ%			{
				debug(9, "", "TK_EQ\n");
				return(TK_EQ);
			}
%EQV%			{
				debug(9, "", "TK_EQV\n");
				return(TK_EQV);
			}
%GE%			{
				debug(9, "", "TK_GE\n");
				return(TK_GE);
			}
%GT%			{
				debug(9, "", "TK_GT\n");
				return(TK_GT);
			}
%LE%			{
				debug(9, "", "TK_LE\n");
				return(TK_LE);
			}
%LT%			{
				debug(9, "", "TK_LT\n");
				return(TK_LT);
			}
%NE%			{
				debug(9, "", "TK_NE\n");
				return(TK_NE);
			}
%NEQV%			{
				debug(9, "", "TK_NEQV\n");
				return(TK_NEQV);
			}
%NOT%			{
				debug(9, "", "TK_NOT\n");
				return(TK_NOT);
			}
%OR%			{
				debug(9, "", "TK_OR\n");
				return(TK_OR);
			}
%TRUE%			{
				debug(9, "", "TK_TRUE\n");
				return(TK_TRUE);
			}
%FALSE%			{
				debug(9, "", "TK_FALSE\n");
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

				debug(9, "", "TK_KEYWORD %s %d %d\n", yytext, yyleng, tv);
				return(tv);
			}

{ULET}+({DIG}|{ULET})*	{ 
				debug(9, "", "TK_NAME %s %d\n", yytext, yyleng);
				return(TK_NAME); 
			}

{UICON}			{
				debug(9, "", "TK_ICON %s %d\n", yytext, yyleng);
				return(TK_ICON); 
			}

{RCON}			{
				debug(9, "", "TK_RCON %s %d\n", yytext, yyleng);
				return(TK_RCON); 
			}

{DCON}			{
				debug(9, "", "TK_DCON %s %d\n", yytext, yyleng);
				return(TK_DCON); 
			}

\'([^\']|(\'\'))*\'	{
				debug(9, "", "TK_SCON %s %d\n", yytext, yyleng);
				return(TK_SCON);
			}
"-"			{
				debug(9, "", "TK_MINUS\n");
				return(TK_MINUS);
			}
"+"			{
				debug(9, "", "TK_PLUS\n");
				return(TK_PLUS);
			}
"/"			{
				debug(9, "", "TK_SLASH\n");
				return(TK_SLASH);
			}
"**"			{
				debug(9, "", "TK_POWER\n");
				return(TK_POWER);
			}
"*"			{
				debug(9, "", "TK_STAR\n");
				return(TK_STAR);
			}
"("			{
				debug(9, "", "TK_LPAR\n");
				return(TK_LPAR);
			}
")"			{
				debug(9, "", "TK_RPAR\n");
				return(TK_RPAR);
			}
","			{
				debug(9, "", "TK_COMMA\n");
				return(TK_COMMA);
			}
":"			{
				debug(9, "", "TK_COLON\n");
				return(TK_COLON);
			}
"="			{
				debug(9, "", "TK_EQUALS\n");
				return(TK_EQUALS);
			}
"$"			{
				debug(9, "", "TK_CURRENCY\n");
				return(TK_CURRENCY);
			}
"//"			{
				debug(9, "", "TK_CONCAT\n");
				return(TK_CONCAT);
			}
\n			{
				debug(9, "", "TK_EOS\n");
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
