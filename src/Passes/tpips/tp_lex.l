EMPTY ("//[^\n]*"|[ \t]+)

%START COMMENT TEXT

%a 10000
%o 30000

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "pipsdbm.h"
#include "ri-util.h"
#include "resources.h"
#include "phases.h"
#include "builder_map.h"
#include "properties.h"
#include "pipsmake.h"

#include "list.h"
#include "ri-util.h"
#include "top-level.h"
#include "tpips.h"
#include "y.tab.h"
%}

%%
<TEXT>{EMPTY}		;
<TEXT>({EMPTY}?\n)+{EMPTY}?      {RETURN SEPARATOR;}
<TEXT>"/*"		{BEGIN COMMENT;}
<TEXT>"//[^\n]*\n"	;
<COMMENT>"*/"		{BEGIN TEXT;}
<COMMENT>(.|\n)      	;
<TEXT>"."               {RETURN DOT;}
<TEXT>"OPEN"|"open"     {RETURN OPEN;}
<TEXT>"CREATE"|"create" {RETURN CREATE;}
<TEXT>"CLOSE"|"close"   {RETURN CLOSE;}
<TEXT>"DELETE"|"delete" {RETURN DELETE;}
<TEXT>"MODULE"|"module" {RETURN MODULE;}
<TEXT>"MAKE"|"make"     {RETURN MAKE;}
<TEXT>"APPLY"|"apply"   {RETURN APPLY;}
<TEXT>"DISPLAY"|"display"       {RETURN DISPLAY;}
<TEXT>"ACTIVATE"|"activate"     {RETURN ACTIVATE;}
<TEXT>"EXIT"|"exit"     {RETURN EXIT;}
<TEXT>"$ALL"            {RETURN OWNER_ALL;}
<TEXT>"$PROGRAM"        {RETURN OWNER_PROGRAM;}
<TEXT>"$MAIN"           {RETURN OWNER_MAIN;}
<TEXT>"$MODULE"         {RETURN OWNER_MODULE;}
<TEXT>"$CALLERS"        {RETURN OWNER_CALLERS;}
<TEXT>"$CALLEES"        {RETURN OWNER_CALLEES;}
<TEXT>[0-9A-Za-z_-]+    {RETURN NAME;}
<TEXT>[\.0-9A-Za-z_-]+  {RETURN FILE_NAME;}
<TEXT>[/\.0-9A-Za-z_-]+ {RETURN PATH_NAME;}
<TEXT,COMMENT>.         {RETURN UNKNOW_CHAR;}
%%
yywrap()
{
    return(1) ;
}

init_lex()
{
	BEGIN TEXT;
}
