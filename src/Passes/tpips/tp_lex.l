%{
/*
 * $Id$
 *
 * $Log: tp_lex.l,v $
 * Revision 1.36  1998/04/29 08:24:27  coelho
 * "show" keyword added.
 *
 * Revision 1.35  1997/12/15 13:49:14  coelho
 * new syntax. () -> [] and $ALL -> %ALL
 *
 * Revision 1.34  1997/12/12 22:17:06  coelho
 * order fix for lonely comments.
 *
 * Revision 1.33  1997/12/12 12:28:39  coelho
 * ! == shell.
 *
 * Revision 1.32  1997/12/11 16:07:48  coelho
 * if unknow, and implicit shell is assumed.
 *
 * Revision 1.31  1997/12/05 16:50:16  coelho
 * checkpoint added.
 *
 * Revision 1.30  1997/12/05 13:29:21  coelho
 * CAPPLY keyword added.
 *
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

/* string to be processed by some other lexer/parser.
 */
static char unk[2]; /* one char lines are not implicit shells */

#define RETURN(x)  pips_debug(8, "init %d\n", x); return x
#define KEYWORD(x) pips_debug(8, "key %d\n", x); BEGIN INITIAL; return x
#define LITTERAL(x) 				\
	pips_debug(8, "lit %d\n", x); 		\
	BEGIN LIT; 				\
	return x
#define RESTART(x) pips_debug(8, "rst %d\n", x); BEGIN KEY; return x

%}

%s KEY STR LIT UNK

%%

<KEY>setenv		{ KEYWORD(TK_SET_ENVIRONMENT); }
<KEY>getenv		{ KEYWORD(TK_GET_ENVIRONMENT); }
<KEY>open		{ KEYWORD(TK_OPEN); }
<KEY>create		{ KEYWORD(TK_CREATE); }
<KEY>close		{ KEYWORD(TK_CLOSE); }
<KEY>checkpoint		{ KEYWORD(TK_CHECKPOINT); }
<KEY>delete		{ KEYWORD(TK_DELETE); }
<KEY>module		{ KEYWORD(TK_MODULE); }
<KEY>make		{ KEYWORD(TK_MAKE); }
<KEY>apply		{ KEYWORD(TK_APPLY); }
<KEY>capply		{ KEYWORD(TK_CAPPLY); }
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
<KEY>show 		{ KEYWORD(TK_SHOW); }

<KEY>setproperty	{ LITTERAL(TK_SET_PROPERTY); }
<KEY>set		{ LITTERAL(TK_SET_PROPERTY); }
<KEY>echo 		{ LITTERAL(TK_ECHO); }
<KEY>shell 		{ LITTERAL(TK_SHELL); }
<KEY>!			{ LITTERAL(TK_SHELL); }

<KEY,INITIAL>[ \t]+	/* skip blanks... */

<UNK>[^\n]*		{ yylval.name = strdup(concatenate(unk, yytext, 0));
			  KEYWORD(TK_UNKNOWN); }

<LIT>[^\n]* 		{ yylval.name = strdup(yytext); RETURN(TK_LINE); }
<INITIAL,KEY>#[^\n]*	{ BEGIN INITIAL;
			  /* skip comments... (not for echo, shell set). */}
<KEY>.			{ unk[0]=yytext[0]; unk[1]='\0'; BEGIN UNK; }

<INITIAL>\(		{ RETURN(TK_OPENPAREN); }
<INITIAL>\)		{ RETURN(TK_CLOSEPAREN); }

<INITIAL>\[		{ RETURN(TK_OPENPAREN); }
<INITIAL>\]		{ RETURN(TK_CLOSEPAREN); }

<INITIAL>\,		{ RETURN(TK_COMMA); }
<INITIAL>= 		{ RETURN(TK_EQUAL); }

<LIT,INITIAL>[\n;]	{ RESTART(TK_ENDOFLINE); }
<LIT,INITIAL><<EOF>>	{ RESTART(TK_ENDOFLINE); }

<STR>[^\"]*\"		{ /* " */ yylval.name = strdup(yytext);
			  yylval.name[strlen(yytext)-1] = '\0';
		  	  BEGIN INITIAL; RETURN(TK_A_STRING);}
<INITIAL>\"		{ /* " */ BEGIN STR;} 

<INITIAL>"$ALL"		{ RETURN(TK_OWNER_ALL); }
<INITIAL>"$PROGRAM"	{ RETURN(TK_OWNER_PROGRAM); }
<INITIAL>"$MAIN"	{ RETURN(TK_OWNER_MAIN); }
<INITIAL>"$MODULE"	{ RETURN(TK_OWNER_MODULE); }
<INITIAL>"$CALLERS"	{ RETURN(TK_OWNER_CALLERS); }
<INITIAL>"$CALLEES"	{ RETURN(TK_OWNER_CALLEES); }

<INITIAL>"%ALL"		{ RETURN(TK_OWNER_ALL); }
<INITIAL>"%PROGRAM"	{ RETURN(TK_OWNER_PROGRAM); }
<INITIAL>"%MAIN"	{ RETURN(TK_OWNER_MAIN); }
<INITIAL>"%MODULE"	{ RETURN(TK_OWNER_MODULE); }
<INITIAL>"%CALLERS"	{ RETURN(TK_OWNER_CALLERS); }
<INITIAL>"%CALLEES"	{ RETURN(TK_OWNER_CALLEES); }

<INITIAL>[/\.0-9A-Za-z_:-]+ 	{ yylval.name=strdup(yytext); RETURN(TK_NAME);}

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
