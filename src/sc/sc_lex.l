%START COMMENT TEXT
%{
/*
	Grammaire lex necessaire pour l'analyse lexicale d'un systeme 
	d'assertions. Les tokens renvoyes sont commentes dans le fichier 
	"sc_gram.y".
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> /* flex uses isatty */

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#include "sc_gram.h"

#define RETURN(x) /* fprintf(stderr, "lex: %d\n", x); */ return x

%}

%%
<TEXT>VAR			{ RETURN(VAR); } 
<TEXT>[A-Za-z_]['A-Za-z0-9_:#]* { yylval.Variable = (Variable) strdup(yytext); /*'*/
                                  RETURN(IDENT); }
<TEXT>[0-9]*			{ sscan_Value(yytext,&yylval.Value);
                                  RETURN(CONSTANTE); }
<TEXT>"=="			{ RETURN(EGAL); }
<TEXT>","			{ RETURN(VIRG); }
<TEXT>"+"			{ RETURN(PLUS); }
<TEXT>"-"			{ RETURN(MOINS); }
<TEXT>">"			{ RETURN(SUP); }
<TEXT>">="			{ RETURN(SUPEGAL); }
<TEXT>"<"			{ RETURN(INF); }
<TEXT>"<="			{ RETURN(INFEGAL); }
<TEXT>"{"			{ RETURN(ACCOUVR); }
<TEXT>"}"			{ RETURN(ACCFERM); }
<TEXT>[ \t\n]*		;
<TEXT>"#"                       {BEGIN COMMENT;}
<COMMENT>\n                     {BEGIN TEXT;}
<COMMENT>[^\n]*         ;
%% 

#ifdef FLEX_SCANNER
int yyinput()
{
    return( input()) ;
}
#endif
 
int yywrap(void)
{
#ifdef FLEX_SCANNER
    yy_init = 1 ;
#endif
    return( 1 ) ;
}
 
void sc_init_lex(void)
{
    /*yyrestart(); */
    BEGIN TEXT;
}

extern boolean yysyntax_error;

int yyerror(char *s)
{
	/* procedure minimun de recouvrement d'erreurs */
	int c;

	(void) fprintf(stderr,"%s near %s\n",s,yytext);
	while ((c = getchar()) != EOF)
		putchar(c);

	yysyntax_error = TRUE;
	return TRUE;
}

