/****************************************************/
/* This file contains the lexer for Cipol-light.    */
/****************************************************/

/*** Include files ***/

%{

#include <stdio.h>
#include <stddef.h>
#include "Common/types.h"
#include "Pip/types.h"
#include "Irisa/types.h"
#include "cipol.h"
#include "erroridt.h"
#include "y.tab.h"

%}

/*** Global value ***/

%{

#define BUFFER_SIZE	1024
char lastToken[BUFFER_SIZE];	/** Keep trace of last read token **/

%}

/*** Macro to simplify the use of ``validToken'' ***/

%{

#define Return(token)					\
	validToken(yytext);				\
	return(token)

%}

/*** Lex constants ***/

SEP	[ \t]
CR	\n
SEPCR	[ \t\n]
DD      [0-9]
STR     \"[^\"]*\"

/*** Tools for basic type conversion ***/

%{

/** This function replace the sub-strings `""` by a `"` ,   **/
/** remove the '\n' characters and put the result string    **/
/** in yylval.                                              **/

void createString(string)
char *string;
{
char *p,*q;
char *newString;
unsigned char quote;

quote=NULL;
p=q=string+1;
while(*p)
{
	if(*p=='"'&&quote){
		p++; quote=NULL;
		}
	else{
		if(*p=='"') quote=1;
		if(*p=='\n') p++;
		else (*q++)=(*p++);
		}
}
*(q-1)='\0';
newString=(char *)malloc((strlen(string)+1)*sizeof(char));
if(!newString) errError(OUT_OF_MEM,TERROR,"createString",NULL);
strcpy(newString,string+1);
yylval.string=newString;
}

/** This function copy a string into the last token buffer **/
/** Each rule must call this routine.                      **/

void validToken(string)
char *string;
{
strncpy(lastToken,string,BUFFER_SIZE);
lastToken[BUFFER_SIZE-1]='\0';
}

%}

%%

;[^\n]*\n				{ errorLine++ ; }
{CR}					{ errorLine++ ; }
{SEP}+					;

"-"?{DD}+				{
					yylval.integer=atoi(yytext);
					Return(INTEGER);
					}
{STR}({SEP}*{STR})*			{
					createString(yytext);
					Return(STRING);
					}

"("					{ Return(BEGIN_FUNCTION); }
")"					{ Return(END_FUNCTION); }
("#["|"[")				{ Return(BEGIN_VECTOR); }
"]"					{ Return(END_VECTOR); }

"print_int"{SEPCR}+			{ Return(INTEGER_PRINT); }
"print_str"{SEPCR}+			{ Return(STRING_PRINT); }
"print_mat"{SEPCR}+			{ Return(MATRIX_PRINT); }
"print_dom"{SEPCR}+			{ Return(DOMAIN_PRINT); }
"print_pip"{SEPCR}+			{ Return(PIP_PRINT); }

"pip"{SEPCR}+				{ Return(PIP); }
"c2d"{SEPCR}+				{ Return(C2D); }
"r2d"{SEPCR}+				{ Return(R2D); }
"p2c"{SEPCR}+				{ Return(P2C); }
"p2r"{SEPCR}+				{ Return(P2R); }
"inter"{SEPCR}+				{ Return(INTER); }
"union"{SEPCR}+				{ Return(UNION); }
"differ"{SEPCR}+			{ Return(DIFFER); }
"convex"{SEPCR}+			{ Return(CONVEX); }
"preimage"{SEPCR}+			{ Return(PRE_IMAGE); }
"image"{SEPCR}+				{ Return(IMAGE); }
"include"{SEPCR}+			{ Return(INCLUDE); }

.					{ Return(LAST_TOKEN+1); }
