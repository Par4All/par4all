EMPTY (("#".*)|([ \\t]+("#".*)?))

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
 result = (c == '\0') ? YY_NULL : (buf[0] = c, 1); \
}

#endif

%}

%s KEY FNAME WNAME RES PHASE PROP

%%
<KEY,FNAME,WNAME,RES,PHASE,PROP>{EMPTY}  {return SEPARATOR;}
<KEY>"open"     {BEGIN WNAME;return OPEN;}
<KEY>"create"   {BEGIN WNAME;return CREATE;}
<KEY>"close"    {BEGIN WNAME;return CLOSE;}
<KEY>"delete"   {BEGIN WNAME;return DELETE;}
<KEY>"module"   {BEGIN WNAME;return MODULE;}
<KEY>"make"     {BEGIN RES;return MAKE;}
<KEY>"apply"    {BEGIN PHASE;return APPLY;}
<KEY>"display"  {BEGIN RES;return DISPLAY;}
<KEY>"activate" {BEGIN PHASE;return ACTIVATE;}
<KEY>"set"      {BEGIN PROP;return SET;}
<KEY>"get"      {BEGIN PROP;return GET;}
<KEY>"info"     {BEGIN WNAME;return INFO;}
<FNAME>[/\\.0-9A-Za-z_-]+       {yylval.name = strdup(yytext);return FILE_NAME;}
<WNAME>[0-9A-Za-z_-]+   {BEGIN FNAME;yylval.name =strdup(yytext);return WORKSPACE;}
<RES,PHASE>\\{    {return OPENBRACE;}
<RES,PHASE>\\.    {return DOT;}
<RES,PHASE>\\,    {return COMMA;}
<RES,PHASE>\\}    {return CLOSEBRACE;}
<RES,PHASE>"$ALL."             {return OWNER_ALL;}
<RES,PHASE>"$PROGRAM."         {return OWNER_PROGRAM;}
<RES,PHASE>"$MAIN."            {return OWNER_MAIN;}
<RES,PHASE>"$MODULE."          {return OWNER_MODULE;}
<RES,PHASE>"$CALLERS."         {return OWNER_CALLERS;}
<RES,PHASE>"$CALLEES."         {return OWNER_CALLEES;}
%%%%%%%%%%
<RES,PHASE>[0-9A-Za-z_-]+      {yylval.name =strdup(yytext);return OWNER_NAME;}
<KEY,PROP>[a-zA-Z0-9_]+        {return UNKNOW_CHAR; /* actually it is chars ... */}
<KEY,FNAME,WNAME,RES,PHASE,PROP>.          {return UNKNOW_CHAR;}
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
