%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "tpips.h"
#include "tp_yacc.h"

#define skip_blanks(str) \
  while (*str && (*str==' ' || *str=='\t' || *str=='\n')) str++

/* now get the characters from a string, not from a file */
#ifndef FLEX_SCANNER

#undef input()
#define input() tpips_lex_input()
#undef unput()
#define unput(c) tpips_lex_unput(c)
void yyrestart(FILE *f){ yyin = f;}

#else

#define YY_INPUT(buf,result,max_size) 			\
{ 							\
 int c = tpips_lex_input(); 				\
 result = (c == '\0') ? YY_NULL : (buf[0] = c, 1); 	\
}

#endif

static char * line_to_parse;
static char * line_parsed;

void tpips_set_line_to_parse(char * line)
{
    skip_blanks(line);

    /* store line pointer */
    line_parsed = line_to_parse = line;

    /* cut line at the beginning of comments */
    while (*line)
    {
	if ((*line) == TPIPS_COMMENT_PREFIX)
	{
	    *line = 0;
	    break;
	}
	line++;
    }
}

char * tpips_get_line_to_parse(void)
{
    return line_to_parse;
}

/*
static void tpips_lex_unput(int c)
{
    pips_debug(9,"unput char '%c'(0x%2x)\n", c,c);
    pips_assert("some place to unput a char", line_parsed<line_to_parse);
    *(--line_to_parse) = (char) c;
}
*/

static void tpips_lex_print_pos(FILE* fout)
{
    fprintf(fout,"%s\n",line_parsed);
    fprintf(fout,"%*s^\n",(int)((long) line_to_parse - (long)line_parsed),"");
}

static int tpips_lex_input (void)
{
    char c = *line_to_parse;
    pips_debug(9,"input char '%c'(0x%2x) from input\n", c, c);
    if (c) line_to_parse++;
    return (int) c;
}

#define KEYWORD(x) BEGIN INITIAL; return x

%}

%s KEY STR

%%

<KEY>setenv		{ KEYWORD(SET_ENVIRONMENT); }
<KEY>getenv		{ KEYWORD(GET_ENVIRONMENT); }
<KEY>open		{ KEYWORD(OPEN); }
<KEY>create		{ KEYWORD(CREATE); }
<KEY>close		{ KEYWORD(CLOSE); }
<KEY>delete		{ KEYWORD(DELETE); }
<KEY>module		{ KEYWORD(MODULE); }
<KEY>make		{ KEYWORD(MAKE); }
<KEY>apply		{ KEYWORD(APPLY); }
<KEY>display 		{ KEYWORD(DISPLAY); }
<KEY>activate 		{ KEYWORD(ACTIVATE); }
<KEY>getproperty	{ KEYWORD(GET_PROPERTY); }
<KEY>get		{ KEYWORD(GET_PROPERTY); }
<KEY>info		{ KEYWORD(INFO); }
<KEY>cd 		{ KEYWORD(CDIR); }
<KEY>pwd 		{ KEYWORD(PWD); }

\(			{ return OPENPAREN; }
\)			{ return CLOSEPAREN; }
\,			{ return COMMA; }
= 			{ return EQUAL; }

<STR>[^\"]*\"		{ yylval.name = strdup(concatenate("\"", yytext, 0));
		  	  BEGIN INITIAL; return A_STRING;}
\"			{ BEGIN STR;} /* " */

"$ALL"			{ return OWNER_ALL; }
"$PROGRAM"		{ return OWNER_PROGRAM; }
"$MAIN"			{ return OWNER_MAIN; }
"$MODULE"		{ return OWNER_MODULE; }
"$CALLERS"		{ return OWNER_CALLERS; }
"$CALLEES"		{ return OWNER_CALLEES; }

[/\.0-9A-Za-z_-]+ { yylval.name = strdup(yytext); return NAME;}
[ \t\n]+		/* skip blanks */
.			{ pips_user_error("unexpected char %c\n", *yytext); }

%%

int yywrap(void){ return 1; }

void yyinit_lex()
{
	BEGIN KEY;
#ifdef FLEX_SCANNER
	yy_init = 1;
#endif
}

void yyerror(char * s)
{
    tpips_init(); /* needed for the user error... */
    tpips_lex_print_pos(stderr);
    pips_user_error("%s\n", s);
}
