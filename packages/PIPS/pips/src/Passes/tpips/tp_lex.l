%option nounput
%option noinput
%{
/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "tpips.h"
#include "tp_yacc.h"

/* disable echoing unmatched characters */
#define ECHO ;

/* now get the characters from a string, not from a file */
#ifndef FLEX_SCANNER

#undef input()
#define input() tpips_lex_input()
#undef unput()
#define unput(c) tpips_lex_unput(c)
void yyrestart(FILE *f){ yyin = f;}

#else

#define YY_INPUT(buf,result,max_size)                  \
  {                                                    \
    int c = tpips_lex_input();                         \
    result = (c == '\0') ? YY_NULL : (buf[0] = c, 1);  \
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
  cr_not_returned = true;
}

char * tpips_get_line_to_parse(void)
{
  return line_to_parse;
}

static int back = 0;

void tpips_lex_print_pos(FILE* fout)
{
  int shift = (int)((long) line_to_parse - (long)line_parsed) - back;
  fprintf(fout,
          "%s\n"
          "%*s^\n"
          "at line %d in \"%s\"\n", line_parsed, shift, "",
          tpips_current_line(), tpips_current_name());
}

static int tpips_lex_input (void)
{
  char c = *line_to_parse;
  pips_debug(9,"input char '%c'(0x%2x) from input\n", c, c);
  if (c) line_to_parse++;
  /* a fake \n is provided, to check for nargs in the syntax... */
  else if (cr_not_returned) { cr_not_returned=false; return (int) '\n'; }
  return (int) c;
}

/* string to be processed by some other lexer/parser.
 */
static char unk[2]; /* one char lines are not implicit shells */

#define RETURN(x)  pips_debug(8, "init %d\n", x); return x
#define KEYWORD(x) pips_debug(8, "key %d\n", x); BEGIN INITIAL; return x
#define LITTERAL(x)                              \
  pips_debug(8, "spc %d\n", x);                  \
  BEGIN SPC;                                     \
  return x
#define RESTART(x) pips_debug(8, "rst %d\n", x); BEGIN KEY; return x

%}

%s KEY STR SPC LIT UNK

%%

<KEY>setenv      { KEYWORD(TK_SET_ENVIRONMENT); }
<KEY>getenv      { KEYWORD(TK_GET_ENVIRONMENT); }
<KEY>unsetenv    { KEYWORD(TK_UNSET_ENVIRONMENT); }
<KEY>open        { KEYWORD(TK_OPEN); }
<KEY>create      { KEYWORD(TK_CREATE); }
<KEY>close       { KEYWORD(TK_CLOSE); }
<KEY>checkpoint  { KEYWORD(TK_CHECKPOINT); }
<KEY>delete      { KEYWORD(TK_DELETE); }
<KEY>module      { KEYWORD(TK_MODULE); }
<KEY>make        { KEYWORD(TK_MAKE); }
<KEY>apply       { KEYWORD(TK_APPLY); }
<KEY>capply      { KEYWORD(TK_CAPPLY); }
<KEY>display     { KEYWORD(TK_DISPLAY); }
<KEY>activate    { KEYWORD(TK_ACTIVATE); }
<KEY>getproperty { KEYWORD(TK_GET_PROPERTY); }
<KEY>get         { KEYWORD(TK_GET_PROPERTY); }
<KEY>info        { KEYWORD(TK_INFO); }
<KEY>cd          { KEYWORD(TK_CDIR); }
<KEY>pwd         { KEYWORD(TK_PWD); }
<KEY>source      { KEYWORD(TK_SOURCE); }
<KEY>quit        { KEYWORD(TK_QUIT); }
<KEY>bye         { KEYWORD(TK_QUIT); }
<KEY>exit        { KEYWORD(TK_EXIT); }
<KEY>help        { KEYWORD(TK_HELP); }
<KEY>remove      { KEYWORD(TK_REMOVE); }
<KEY>show        { KEYWORD(TK_SHOW); }
<KEY>checkactive { KEYWORD(TK_CHECKACTIVE); }
<KEY>version     { KEYWORD(TK_VERSION); }
<KEY>touch       { KEYWORD(TK_TOUCH); }

<KEY>setproperty { LITTERAL(TK_SET_PROPERTY); }
<KEY>setprop     { LITTERAL(TK_SET_PROPERTY); }
<KEY>set         { LITTERAL(TK_SET_PROPERTY); }
<KEY>echo        { LITTERAL(TK_ECHO); }
<KEY>shell       { LITTERAL(TK_SHELL); }
<KEY>sh          { LITTERAL(TK_SHELL); }
<KEY>!           { LITTERAL(TK_SHELL); }

<KEY,INITIAL>[ \t]+  /* skip blanks... */

<UNK>[^\r\n]*    { yylval.name = strdup(concatenate(unk, yytext, NULL));
   KEYWORD(TK_UNKNOWN); }

<SPC>[ \t]+      { BEGIN LIT; }
<SPC>[^ \t\r\n][^\r\n]* {
   back = strlen(yytext);
   yyerror("space expected after initial keyword");
   back = 0;
 }
<LIT>[^\r\n]* { yylval.name = strdup(yytext); RETURN(TK_LINE); }
<INITIAL,KEY>#[^\r\n]*  { BEGIN INITIAL;
   /* skip comments... (not for echo, shell set). */}
<KEY>.      { unk[0]=yytext[0]; unk[1]='\0'; BEGIN UNK; }

<INITIAL>\(    { RETURN(TK_OPENPAREN); }
<INITIAL>\)    { RETURN(TK_CLOSEPAREN); }

<INITIAL>\[    { RETURN(TK_OPENPAREN); }
<INITIAL>\]    { RETURN(TK_CLOSEPAREN); }

<INITIAL>\,    { RETURN(TK_COMMA); }
<INITIAL>=     { RETURN(TK_EQUAL); }

<LIT,INITIAL,SPC>(\n|\r|\r\n|;)  { RESTART(TK_ENDOFLINE); }
<LIT,INITIAL><<EOF>>         { RESTART(TK_ENDOFLINE); }

<STR>[^\"]*\"  {
  // string contents without surrounding double quotes
  // no attempt at managing escaping double quotes
  yylval.name = strdup(yytext);
  yylval.name[strlen(yytext)-1] = '\0';
  BEGIN INITIAL; RETURN(TK_A_STRING);}
<INITIAL>\"    { BEGIN STR; }

<INITIAL>"$ALL"      { RETURN(TK_OWNER_ALL); }
<INITIAL>"$ALLFUNC"  { RETURN(TK_OWNER_ALLFUNC); }
<INITIAL>"$PROGRAM"  { RETURN(TK_OWNER_PROGRAM); }
<INITIAL>"$MAIN"     { RETURN(TK_OWNER_MAIN); }
<INITIAL>"$MODULE"   { RETURN(TK_OWNER_MODULE); }
<INITIAL>"$CALLERS"  { RETURN(TK_OWNER_CALLERS); }
<INITIAL>"$CALLEES"  { RETURN(TK_OWNER_CALLEES); }

<INITIAL>"%ALL"      { RETURN(TK_OWNER_ALL); }
<INITIAL>"%ALLFUNC"  { RETURN(TK_OWNER_ALLFUNC); }
<INITIAL>"%PROGRAM"  { RETURN(TK_OWNER_PROGRAM); }
<INITIAL>"%MAIN"     { RETURN(TK_OWNER_MAIN); }
<INITIAL>"%MODULE"   { RETURN(TK_OWNER_MODULE); }
<INITIAL>"%CALLERS"  { RETURN(TK_OWNER_CALLERS); }
<INITIAL>"%CALLEES"  { RETURN(TK_OWNER_CALLEES); }

<INITIAL>[/\.0-9A-Za-z_:-][!/\.0-9A-Za-z_:-]*  {
  yylval.name=strdup(yytext);
  RETURN(TK_NAME);
 }
.      { pips_user_warning("Unrecognized character %s\n",yytext); }

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
