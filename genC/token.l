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


/* token.l 

   When a new keyword is added, update the KEYWORDS array in build.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "z.tab.h"

#ifdef FLEX_SCANNER
#undef yywrap
#endif

extern int Current_start, Current_first, Current_index ;

%}

%%
external		{return( GRAM_EXTERNAL );}
import                  {return( GRAM_IMPORT );}
tabulated		{return( TABULATED );}
from			{return( FROM );}
persistant              {return( PERSISTANT );}
persistent              {return( PERSISTANT );}
\,			{return( COMMA );}
\:			{return( COLUMN );}
\;			{return( SEMI_COLUMN );}
x			{return( AND );}
\+			{return( OR );}
"->"                    {return( ARROW );}
\*			{return( STAR );}
\[			{return( LB );}
\]			{return( RB );}
\{			{return( LR );}
\}			{return( RR );}
\"[_a-zA-Z0-9/\.~]+\"	{return( GRAM_FILE );}
=			{return( EQUAL );}
^--NEWGEN-START\ [0-9]*\n {
                        Current_start = 
                             atoi( yytext+strlen( "--NEWGEN-START " )) ;
			}
^--NEWGEN-FIRST\ [0-9]*\n {
                        Current_first = 
                             atoi( yytext+strlen( "--NEWGEN-FIRST " )) ;
                        }
^--NEWGEN-INDEX\ [0-9]*\n {
                        Current_index = 
                             atoi( yytext+strlen( "--NEWGEN-INDEX " )) ;
                        }
--.*\n			{}
[_a-zA-Z][_a-zA-Z0-9]*	{return( IDENT );}
[0-9]+                  {return( GRAM_INT );}
[ \t\n]*		;
%%

int yywrap()
{
#ifdef FLEX_SCANNER
    yy_init = 1 ;
#endif
    return( 1 ) ;
}

#ifdef FLEX_SCANNER
int yyinput()
{
    return( input()) ;
}
#endif
