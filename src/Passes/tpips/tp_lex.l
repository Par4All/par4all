EMPTY (([ \\t\\n])+)
OPTEMPTY (([ \\t\\n])*)

%e 100000
%p 100000
%n 100000
%k 100000
%a 100000
%o 100000

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"


#include "misc.h"

/*
#include "ri.h"
#include "database.h"
#include "makefile.h"
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "builder_map.h"
#include "properties.h"
#include "pipsmake.h"
#include "ri-util.h"
#include "top-level.h"
*/

#include "list.h"
#include "tpips.h"
#include "y.tab.h"


/* now get the characters from a string, not from a file */
#ifndef FLEX_SCANNER

#undef input()
#define input() tpips_lex_input()
#undef unput()
#define unput(c) tpips_lex_unput(c)

#else

#define YY_INPUT(buf,result,max_size) \
{ \
 int c = tpips_lex_input(); \
 result = (c == '\\0') ? YY_NULL : (buf[0] = c, 1); \
}

#endif

%}

%s KEY FNAME WNAME RES PHASE PROP SVAL OWN

%%
<KEY>{OPTEMPTY}"open" {
    debug(8,"yylex","read token OPEN\\n");
    BEGIN WNAME;
    return OPEN;
}

<KEY>{OPTEMPTY}"create" {
    debug(8,"yylex","read token CREATE\\n");
    BEGIN WNAME;
    return CREATE;
}

<KEY>{OPTEMPTY}"close" {
    debug(8,"yylex","read token CLOSE\\n");
    BEGIN WNAME;
    return CLOSE;
}

<KEY>{OPTEMPTY}"delete" {
    debug(8,"yylex","read token DELETE\\n");
    BEGIN WNAME;
    return DELETE;
}

<KEY>{OPTEMPTY}"module"	{
    debug(8,"yylex","read token MODULE\\n");
    BEGIN WNAME;
    return MODULE;
}

<KEY>{OPTEMPTY}"make" {
    debug(8,"yylex","read token MAKE\\n");
    BEGIN RES;
    return MAKE;
}

<KEY>{OPTEMPTY}"apply" {
    debug(8,"yylex","read token APPLY\\n");
    BEGIN PHASE;
    return APPLY;
}

<KEY>{OPTEMPTY}"display" {
    debug(8,"yylex","read token DISPLAY\\n");
    BEGIN RES;
    return DISPLAY;
}

<KEY>{OPTEMPTY}"activate" {
    debug(8,"yylex","read token ACTIVATE\\n");
    BEGIN PHASE;
    return ACTIVATE;
}

<KEY>{OPTEMPTY}"set" {
    debug(8,"yylex","read token SET_PROPERTY\\n");
    BEGIN PROP;
    return SET_PROPERTY;
}

<KEY>{OPTEMPTY}"get" {
    debug(8,"yylex","read token GET_PROPERTY\\n");
    BEGIN PROP;
    return GET_PROPERTY;
}

<KEY>{OPTEMPTY}"info" {
    debug(8,"yylex","read token INFO\\n");
    BEGIN WNAME;
    return INFO;
}

<KEY,FNAME,WNAME,RES,PHASE,PROP,SVAL,OWN>{EMPTY} {
    debug(8,"yylex","read token SEPARATOR\\n");
    return SEPARATOR;
}

<FNAME>[/\\.0-9A-Za-z_-]+ {
    debug(8,"yylex","read token FILE_NAME\\n");
    yylval.name = strdup(yytext);
    return FILE_NAME;
}

<WNAME>[0-9A-Za-z_-]+ {
    debug(8,"yylex","read token WORKSPACE\\n");
    BEGIN FNAME;
    yylval.name = strdup(yytext);
    return WORKSPACE;
}

<RES,PHASE>\\( {
    debug(8,"yylex","read token OPENPAREN\\n");
    BEGIN OWN;
    return OPENPAREN;
}

<OWN>\\) {
    debug(8,"yylex","read token CLOSEPAREN\\n");
    return CLOSEPAREN;
}

<OWN>\\, {
    debug(8,"yylex","read token COMMA\\n");
    return COMMA;
}

<OWN>"$ALL" {
    debug(8,"yylex","read token OWNER_ALL\\n");
    return OWNER_ALL;
}

<OWN>"$PROGRAM" {
    debug(8,"yylex","read token OWNER_PROGRAM\\n");
    return OWNER_PROGRAM;
}

<OWN>"$MAIN" {
    debug(8,"yylex","read token OWNER_MAIN\\n");
    return OWNER_MAIN;
}

<OWN>"$MODULE" {
    debug(8,"yylex","read token OWNER_MODULE\\n");
    return OWNER_MODULE;
}

<OWN>"$CALLERS" {
    debug(8,"yylex","read token OWNER_CALLERS\\n");
    return OWNER_CALLERS;
}

<OWN>"$CALLEES" {
    debug(8,"yylex","read token OWNER_CALLEES\\n");
    return OWNER_CALLEES;
}

%%%%%%%%%%
<OWN>[0-9A-Za-z_-]+ {
    debug(8,"yylex","read token OWNER_NAME\\n");
    yylval.name =strdup(yytext);
    return OWNER_NAME;
}

<KEY,PROP>[a-zA-Z0-9_]+ {
    debug(8,"yylex","read token UNKNOWN_CHAR\\n");
    return UNKNOWN_CHAR;
}

<SVAL>"TRUE" {
    debug(8,"yylex","read token SETVALUE\\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>"FALSE" {
    debug(8,"yylex","read token SETVALUE\\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>[0-9]+ {
    debug(8,"yylex","read token SETVALUE\\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>\\"[^\\"]*\\" {
    debug(8,"yylex","read token SETVALUE\\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<KEY,FNAME,WNAME,RES,PHASE,PROP,OWN>. {
    debug(8,"yylex","read token UNKNOWN_CHAR\\n");
    return UNKNOWN_CHAR;
}

%%

yywrap()
{return 1;}

void yyinit_lex()
{
	BEGIN KEY;
#ifdef FLEX_SCANNER
	yy_init = 1;
#endif
}

void yybegin_key()
{BEGIN KEY;}

void yybegin_fname()
{BEGIN FNAME;}
