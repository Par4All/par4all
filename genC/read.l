%{
/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


/* read.l

   The syntax of tokens written by GEN_WRITE. */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "genC.h"
#include "newgen_include.h"
#include "genread.h"

#ifdef FLEX_SCANNER
#undef yywrap
#endif

#define LITERAL_SIZE 30000

char literal[ LITERAL_SIZE ] = { '\0' } ;

static void append_to_literal() ;

%}

%Start a_string

%%
<a_string>\\\\            {strcat( literal, "\\" );}
<a_string>\\\"		{strcat( literal, "\"" );}
<a_string>[^\\"\n]*       {append_to_literal();}
<a_string>[ \t\n]*        {append_to_literal();}
<a_string>\"		{
			BEGIN 0 ;
			return READ_STRING ;
			}

\[                      {return LB ;}
\#\(\#\]type            {return CHUNK_BEGIN ;}
\#\(			{return VECTOR_BEGIN ;}
\#\]arrow               {return ARROW_BEGIN ;}
\(		        {return LP ;}
\)			{return RP ;}
\{		        {return LC ;}
\}			{return RC ;}
\#\]shared              {return SHARED_POINTER  ;}
\#\]unit		{return READ_UNIT ;}
\#\]bool		{return READ_BOOL ;}
\#\]external 		{return READ_EXTERNAL ;}
\#\]def			{return READ_DEF ;}
\#\]ref			{return READ_REF ;}
\#\]null		{return READ_NULL ;}
\#\]list                {return READ_LIST_UNDEFINED ;}
\#\]set                 {return READ_SET_UNDEFINED ;}
\#\]array               {return READ_ARRAY_UNDEFINED ;}
\#\\.			{ yylval.c=*(yytext+2); return READ_CHAR ;}
[+-]?[ \t]*[0-9]+       { yylval.val=atoi(yytext); return READ_INT ;}
[+-]?[ \t]*[0-9]+\.[0-9]+ { yylval.d=atof(yytext);
                        return READ_FLOAT ;}
\"			{
			BEGIN a_string ;
                        literal[0] = '\0';
			}
[ \t\n]*	        ;
.                       {fatal( "gen_read: Unknown character %x\n", 
				 (int) yytext[0]);}
%%

#ifdef FLEX_SCANNER
int yyinput()
{
    return( input()) ;
}
#endif

yywrap()
{
#ifdef FLEX_SCANNER
    yy_init = 1 ;
#endif
    return( 1 ) ;
}

static void append_to_literal()
{
    int total = strlen( literal )+strlen( yytext ) ;

    if( total < LITERAL_SIZE ) {
        strcat( literal, yytext );
    }
    else { 
        fatal( "read: Not enough space %s\n", 
              itoa( total )) ;
    }
}
