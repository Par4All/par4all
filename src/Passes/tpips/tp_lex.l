EMPTY (([ \t\n])+)
OPTEMPTY (([ \t\n])*)

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

#define YY_INPUT(buf,result,max_size) \
{ \
 int c = tpips_lex_input(); \
 result = (c == '\0') ? YY_NULL : (buf[0] = c, 1); \
}

#endif

%}

%s KEY FNAME WNAME RES PHASE PROP SVAL OWN

%%
<KEY>{OPTEMPTY}"open" {
    pips_debug(8,"read token OPEN\n");
    BEGIN WNAME;
    return OPEN;
}

<KEY>{OPTEMPTY}"create" {
    pips_debug(8,"read token CREATE\n");
    BEGIN WNAME;
    return CREATE;
}

<KEY>{OPTEMPTY}"close" {
    pips_debug(8,"read token CLOSE\n");
    BEGIN WNAME;
    return CLOSE;
}

<KEY>{OPTEMPTY}"delete" {
    pips_debug(8,"read token DELETE\n");
    BEGIN WNAME;
    return DELETE;
}

<KEY>{OPTEMPTY}"module"	{
    pips_debug(8,"read token MODULE\n");
    BEGIN WNAME;
    return MODULE;
}

<KEY>{OPTEMPTY}"make" {
    pips_debug(8,"read token MAKE\n");
    BEGIN RES;
    return MAKE;
}

<KEY>{OPTEMPTY}"apply" {
    pips_debug(8,"read token APPLY\n");
    BEGIN PHASE;
    return APPLY;
}

<KEY>{OPTEMPTY}"display" {
    pips_debug(8,"read token DISPLAY\n");
    BEGIN RES;
    return DISPLAY;
}

<KEY>{OPTEMPTY}"activate" {
    pips_debug(8,"read token ACTIVATE\n");
    BEGIN PHASE;
    return ACTIVATE;
}

<KEY>{OPTEMPTY}"getproperty" {
    pips_debug(8,"read token GET_PROPERTY\n");
    BEGIN PROP;
    return GET_PROPERTY;
}

<KEY>{OPTEMPTY}"get" {
    pips_debug(8,"read token GET_PROPERTY\n");
    BEGIN PROP;
    return GET_PROPERTY;
}

<KEY>{OPTEMPTY}"info" {
    pips_debug(8,"read token INFO\n");
    BEGIN WNAME;
    return INFO;
}

<KEY,FNAME,WNAME,RES,PHASE,PROP,SVAL,OWN>{EMPTY} {
    pips_debug(8,"read token SEPARATOR\n");
    return SEPARATOR;
}

<FNAME>[/\.0-9A-Za-z_-]+ {
    pips_debug(8,"read token FILE_NAME\n");
    yylval.name = strdup(yytext);
    return FILE_NAME;
}

<WNAME>[0-9A-Za-z_-]+ {
    pips_debug(8,"read token WORKSPACE\n");
    BEGIN FNAME;
    yylval.name = strdup(yytext);
    return WORKSPACE;
}

<RES,PHASE>\( {
    pips_debug(8,"read token OPENPAREN\n");
    BEGIN OWN;
    return OPENPAREN;
}

<OWN>\) {
    pips_debug(8,"read token CLOSEPAREN\n");
    return CLOSEPAREN;
}

<OWN>\, {
    pips_debug(8,"read token COMMA\n");
    return COMMA;
}

<OWN>"$ALL" {
    pips_debug(8,"read token OWNER_ALL\n");
    return OWNER_ALL;
}

<OWN>"$PROGRAM" {
    pips_debug(8,"read token OWNER_PROGRAM\n");
    return OWNER_PROGRAM;
}

<OWN>"$MAIN" {
    pips_debug(8,"read token OWNER_MAIN\n");
    return OWNER_MAIN;
}

<OWN>"$MODULE" {
    pips_debug(8,"read token OWNER_MODULE\n");
    return OWNER_MODULE;
}

<OWN>"$CALLERS" {
    pips_debug(8,"read token OWNER_CALLERS\n");
    return OWNER_CALLERS;
}

<OWN>"$CALLEES" {
    pips_debug(8,"read token OWNER_CALLEES\n");
    return OWNER_CALLEES;
}

%%%%%%%%%%
<OWN>[0-9A-Za-z_-]+ {
    pips_debug(8,"read token OWNER_NAME\n");
    yylval.name =strdup(yytext);
    return OWNER_NAME;
}

<KEY,PROP>[a-zA-Z0-9_]+ {
    pips_debug(8,"read token UNKNOWN_CHAR\n");
    return UNKNOWN_CHAR;
}

<SVAL>"TRUE" {
    pips_debug(8,"read token SETVALUE\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>"FALSE" {
    pips_debug(8,"read token SETVALUE\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>[0-9]+ {
    pips_debug(8,"read token SETVALUE\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<SVAL>\"[^\"]*\" {
    pips_debug(8,"read token SETVALUE\n");
    yylval.name = strdup(yytext);
    return SETVALUE;
}

<KEY,FNAME,WNAME,RES,PHASE,PROP,OWN>. {
    pips_debug(8,"read token UNKNOWN_CHAR\n");
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
