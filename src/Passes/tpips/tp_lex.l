%{
/*
 * $Id$
 *
 * $Log: tp_lex.l,v $
 * Revision 1.29  1997/11/27 12:12:59  coelho
 * ; as valid separator.
 *
 */
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

static bool cr_not_returned;
static char * line_to_parse;
static char * line_parsed;

void tpips_set_line_to_parse(char * line)
{
    skip_blanks(line);
    /* store line pointer */
    line_parsed = line_to_parse = line;
    cr_not_returned = TRUE;
}

char * tpips_get_line_to_parse(void)
{
    return line_to_parse;
}

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
    /* a fake \n is provided, to check for nargs in the syntax... */
    else if (cr_not_returned) { cr_not_returned=FALSE; return (int) '\n'; }
    return (int) c;
}

#define RETURN(x)  pips_debug(8, "init %d\n", x); return x
#define KEYWORD(x) pips_debug(8, "key %d\n", x); BEGIN INITIAL; return x
#define SPECIAL(x) pips_debug(8, "spe %d\n", x); BEGIN SPEC; return x
#define RESTART(x) pips_debug(8, "rst %d\n", x); BEGIN KEY; return x

%}

%s KEY STR SPEC

%%

<KEY>setenv		{ KEYWORD(TK_SET_ENVIRONMENT); }
<KEY>getenv		{ KEYWORD(TK_GET_ENVIRONMENT); }
<KEY>open		{ KEYWORD(TK_OPEN); }
<KEY>create		{ KEYWORD(TK_CREATE); }
<KEY>close		{ KEYWORD(TK_CLOSE); }
<KEY>delete		{ KEYWORD(TK_DELETE); }
<KEY>module		{ KEYWORD(TK_MODULE); }
<KEY>make		{ KEYWORD(TK_MAKE); }
<KEY>apply		{ KEYWORD(TK_APPLY); }
<KEY>display 		{ KEYWORD(TK_DISPLAY); }
<KEY>activate 		{ KEYWORD(TK_ACTIVATE); }
<KEY>getproperty	{ KEYWORD(TK_GET_PROPERTY); }
<KEY>get		{ KEYWORD(TK_GET_PROPERTY); }
<KEY>info		{ KEYWORD(TK_INFO); }
<KEY>cd 		{ KEYWORD(TK_CDIR); }
<KEY>pwd 		{ KEYWORD(TK_PWD); }
<KEY>source		{ KEYWORD(TK_SOURCE); }
<KEY>quit 		{ KEYWORD(TK_QUIT); }
<KEY>exit 		{ KEYWORD(TK_EXIT); }
<KEY>help 		{ KEYWORD(TK_HELP); }
<KEY>remove 		{ KEYWORD(TK_REMOVE); }

<KEY>setproperty	{ SPECIAL(TK_SET_PROPERTY); }
<KEY>set		{ SPECIAL(TK_SET_PROPERTY); }
<KEY>echo 		{ SPECIAL(TK_ECHO); }
<KEY>shell 		{ SPECIAL(TK_SHELL); }

<SPEC>[^\n]* 		{ yylval.name = strdup(yytext); RETURN(TK_LINE); }
<INITIAL,KEY>#[^\n]*	{ BEGIN INITIAL;
			  /* skip comments... (not for echo, shell set). */}

\(			{ RETURN(TK_OPENPAREN); }
\)			{ RETURN(TK_CLOSEPAREN); }
\,			{ RETURN(TK_COMMA); }
= 			{ RETURN(TK_EQUAL); }

<SPEC,INITIAL>[\n;]	{ RESTART(TK_ENDOFLINE); }
<SPEC,INITIAL><<EOF>>	{ RESTART(TK_ENDOFLINE); }

<STR>[^\"]*\"		{ /* " */ yylval.name = strdup(yytext);
			  yylval.name[strlen(yytext)-1] = '\0';
		  	  BEGIN INITIAL; RETURN(TK_A_STRING);}
\"			{ /* " */ BEGIN STR;} 

"$ALL"			{ RETURN(TK_OWNER_ALL); }
"$PROGRAM"		{ RETURN(TK_OWNER_PROGRAM); }
"$MAIN"			{ RETURN(TK_OWNER_MAIN); }
"$MODULE"		{ RETURN(TK_OWNER_MODULE); }
"$CALLERS"		{ RETURN(TK_OWNER_CALLERS); }
"$CALLEES"		{ RETURN(TK_OWNER_CALLEES); }

[/\.0-9A-Za-z_:-]+ 	{ yylval.name = strdup(yytext); RETURN(TK_NAME);}
<KEY,INITIAL>[ \t]+	/* skip blanks... */
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
    tpips_init(); /* needed for user error... */
    tpips_lex_print_pos(stderr);
    pips_user_error("%s\n", s);
}
