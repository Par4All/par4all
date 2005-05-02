/******************************************************/
/* This file contains the grammar for Cipol-light.    */
/******************************************************/

/*** Include files ***/

%{

#include <stdio.h>
#include <stddef.h>
#include "Common/types.h"
#include "Pip/types.h"
#include "Irisa/types.h"
#include "Irisa/polyhedron.h"
#include "cipol.h"
#include "erroridt.h"

#ifdef DBMALLOC4
#	include "malloc4.h"
#endif
#ifdef DBMALLOC9
#	include "malloc9.h"
#endif

%}

/*** Imported value from lex ***/

%{

extern char lastToken[];

%}

/*** C code ***/

%{

/*** Main function ***/

void main()
{
Cipol_Init();		/* A few initializations */
yyparse();		/* The parser do all the work ! */
}

/*** Handle syntax error ***/

#define BUFFER_SIZE	1024
#define TOKEN_MAX_SIZE	12

void yyerror()
{
char buffer[BUFFER_SIZE];

if(strlen(lastToken)>TOKEN_MAX_SIZE){
	lastToken[TOKEN_MAX_SIZE]='\0';
	strcat(lastToken," ...");
	}
sprintf(buffer,"at line %d near ``%s''",errorLine,lastToken);
errError(SYNTAX_ERROR,TWARNING,"parser",buffer);
raiseError();
}

%}

/*** Types used by the Cipol-light tools ***/

%union{
	int		integer;		/* Integer numbers */
	char		*string;		/* Characters strings */
	Matrix		*matrix;		/* Integer matrix */
	Polyhedron	*domain;		/* Convex domain */
	struct pipList	*pipList;		/* PIP result quast */
	}

/*** Tokens for Cipol-light separators ***/

%token			BEGIN_FUNCTION
%token			END_FUNCTION
%token			BEGIN_VECTOR
%token			END_VECTOR

/*** Tokens for Cipol-light basic types  ***/

%token	<integer>	INTEGER
%token	<string>	STRING

/*** Tokens for Cipol-light functions ***/

%token			INTEGER_PRINT
%token			STRING_PRINT
%token			MATRIX_PRINT
%token			DOMAIN_PRINT
%token			PIP_PRINT

%token			PIP
%token			C2D
%token			R2D
%token			P2C
%token			P2R
%token			INTER
%token			UNION
%token			DIFFER
%token			CONVEX
%token			PRE_IMAGE
%token			IMAGE
%token			INCLUDE

/*** Last token (to produce errors) ***/

%token 			LAST_TOKEN

/*** Types for legal objects productions ***/

%type	<integer>	integer
%type	<string>	string
%type	<matrix>	matrix
%type	<domain>	domain
%type	<pipList>	pipList

/*** Types for intermediary productions ***/

%type	<matrix>	partialMatrix,MATRIX

/*** Only expressions with type void are allowed as top-level expressions ***/
/*** These expressions should be a call to a pretty-print function.       ***/

%start voidLoop

%%

/*** Main production ***/

voidLoop:
	|	voidLoop { resetError(); } void
	;

/*** Pretty-printers ***/

void:
		BEGIN_FUNCTION INTEGER_PRINT integer END_FUNCTION
			{ catcherror(printf("%d\n",$3),NULL)
			  fflush(stdout);
			  }
	|	BEGIN_FUNCTION STRING_PRINT string END_FUNCTION
			{ catcherror(printf("%s\n",$3),NULL)
			  if($3) free($3);
			  fflush(stdout);
			  }
	|	BEGIN_FUNCTION MATRIX_PRINT matrix END_FUNCTION
			{ catcherror(Matrix_Pretty($3),NULL)
			  if($3) Matrix_Free($3);
			  fflush(stdout);
			  }
	|	BEGIN_FUNCTION DOMAIN_PRINT domain END_FUNCTION
			{ catcherror(Domain_Pretty($3),NULL)
			  if($3) Domain_Free($3);
			  fflush(stdout);
			  }
	|	BEGIN_FUNCTION PIP_PRINT pipList END_FUNCTION
			{ catcherror(pipListDecompile($3),NULL)
			  if($3) free($3);
			  fflush(stdout);
			  } 
	|	error END_FUNCTION
			{ yyerrok; yyclearin;
			  fflush(stdout);
			  } 
	;

/*** Integer functions (tools which return an integer) ***/

integer:
		INTEGER			{ $$=$1; }
	|	BEGIN_FUNCTION INCLUDE domain domain END_FUNCTION
			{ catcherror(
				$$=PolyhedronIncludes($3,$4),
				$$=NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Domain_Free($4);}
	;

/*** String functions (tools which return a characters string) ***/

string:	
		STRING			{ $$=$1; }
	; 

/*** Matrix functions (tools which return a matrix) ***/

matrix:
		MATRIX			{ $$=$1; }
	|	BEGIN_FUNCTION P2C domain END_FUNCTION
			{ catcherror(
				$$=Polyhedron2Constraints($3),
				$$=(Matrix *)NULL)
			  if ($3) Domain_Free($3);}
	|	BEGIN_FUNCTION P2R domain END_FUNCTION
			{ catcherror(
				$$=Polyhedron2Rays($3),
				$$=NULL)
			  if ($3) Domain_Free($3);}
	;

/*** Domain functions (tools which return a domain or a polyhedron) ***/

domain:
		BEGIN_FUNCTION C2D matrix END_FUNCTION
			{ catcherror(
				$$=Constraints2Polyhedron($3,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Matrix_Free($3);}
	|	BEGIN_FUNCTION R2D matrix END_FUNCTION
			{ catcherror(
				$$=Rays2Polyhedron($3,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Matrix_Free($3);}
	|	BEGIN_FUNCTION INTER domain domain END_FUNCTION
	{
			catcherror(
				$$=DomainIntersection($3,$4,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Domain_Free($4);}
	|	BEGIN_FUNCTION UNION domain domain END_FUNCTION
			{ catcherror(
				$$=DomainUnion($3,$4,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Domain_Free($4);}
	|	BEGIN_FUNCTION DIFFER domain domain END_FUNCTION
			{ catcherror(
				$$=DomainDifference($3,$4,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Domain_Free($4);}
	|	BEGIN_FUNCTION CONVEX domain END_FUNCTION
			{ catcherror(
				$$=DomainConvex($3,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);}
	|	BEGIN_FUNCTION PRE_IMAGE domain matrix END_FUNCTION
			{ catcherror(
				$$=DomainPreimage($3,$4,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Matrix_Free($4);}
	|	BEGIN_FUNCTION IMAGE domain matrix END_FUNCTION
			{ catcherror(
				$$=DomainImage($3,$4,CIPOL_MAX_RAYS),
				$$=(Polyhedron *)NULL)
			  if ($3) Domain_Free($3);
			  if ($4) Matrix_Free($4);}
	;

/*** PipList functions (tools which return a quast) ***/

pipList:
		BEGIN_FUNCTION PIP
			integer integer integer
			integer integer integer
			matrix matrix
		END_FUNCTION
			{ catcherror(
				$$=fpip($3,$4,$5,$6,$7,$8,$9,$10),
				$$=fpipError())
			  if($9) Matrix_Free($9);
			  if($10) Matrix_Free($10);}
	;

/*** Productions for reading intrinsic objects ***/

/** In order to read a matrix **/

MATRIX:
		partialMatrix END_VECTOR END_VECTOR
			{ catcherror(
				$$=Matrix_Close($1),
				$$=(Matrix *)NULL)}
	|	BEGIN_VECTOR END_VECTOR
			{ catcherror(
				$$=Matrix_Create(),
				$$=(Matrix *)NULL)}
	;

partialMatrix:
		BEGIN_VECTOR BEGIN_VECTOR
			{ catcherror(
				$$=Matrix_NewRow(Matrix_Create()),
				$$=(Matrix *)NULL
				)}
	|	partialMatrix END_VECTOR BEGIN_VECTOR
			{ catcherror(
				$$=Matrix_NewRow($1),
				$$=(Matrix *)NULL)}
	|	partialMatrix integer 
			{ catcherror(
				$$=Matrix_AddElement($1,$2),
				$$=(Matrix *)NULL)}
	;
