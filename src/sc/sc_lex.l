%START COMMENT TEXT
%{
/*
	Grammaire lex necessaire pour l'analyse lexicale d'un systeme 
	d'assertions. Les tokens renvoyes sont commentes dans le fichier 
	"sc_gram.y".
*/

#include <stdio.h>
#include "y.tab.h"
%}

%%
<TEXT>VAR			{ return(VAR); }
<TEXT>[A-Za-z_]['A-Za-z0-9_:#]* { return(IDENT); }
<TEXT>[0-9]*			{ return(CONSTANTE); }
<TEXT>"=="			{ return(EGAL); }
<TEXT>","			{ return(VIRG); }
<TEXT>"+"			{ return(PLUS); }
<TEXT>"-"			{ return(MOINS); }
<TEXT>">"			{ return(SUP); }
<TEXT>">="			{ return(SUPEGAL); }
<TEXT>"<"			{ return(INF); }
<TEXT>"<="			{ return(INFEGAL); }
<TEXT>"{"			{ return(ACCOUVR); }
<TEXT>"}"			{ return(ACCFERM); }
<TEXT>[ \t\n]*		;
<TEXT>"#"                       {BEGIN COMMENT;}
<COMMENT>\n                     {BEGIN TEXT;}
<COMMENT>[^\n]*         ;
%%

yywrap()
{
	return(-1);
}

sc_init_lex()
{
        BEGIN TEXT;
}
