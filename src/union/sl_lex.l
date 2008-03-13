
%START COMMENT TEXT
%{
/*
    Grammaire lex necessaire pour l'analyse lexicale d'une liste de systemes.
    Les tokens renvoyes sont commentes dans le fichier "sl_gram.y".
*/

#include <stdio.h>
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#include "y.tab.h"
#ifdef SCAN_VIEW
#define RETURN(token,string) fprintf(stderr,"%s ", string) ; \
                             return(token)
#else
#define RETURN(token,string) return(token)
#endif
%}

%%
<TEXT>VAR                       { RETURN(VAR, "VAR "); }
<TEXT>[A-Za-z_]['A-Za-z0-9_:#]* { RETURN(IDENT, "IDENT "); }
<TEXT>[0-9]*                    { RETURN(CONSTANTE, "CONSTANTE "); }
<TEXT>"=="                      { RETURN(EGAL, "EGAL "); }
<TEXT>","                       { RETURN(VIRG, "VIRG "); }
<TEXT>"+"                       { RETURN(PLUS, "PLUS "); }
<TEXT>"-"                       { RETURN(MOINS, "MOINS "); }
<TEXT>">"                       { RETURN(SUP, "SUP "); }
<TEXT>">="                      { RETURN(SUPEGAL, "SUPEGAL "); }
<TEXT>"<"                       { RETURN(INF, "INF "); }
<TEXT>"<="                      { RETURN(INFEGAL, "INFEGAL "); }
<TEXT>"{"                       { RETURN(ACCOUVR, "ACCOUVR "); }
<TEXT>"}"                       { RETURN(ACCFERM, "ACCFERM "); }
<TEXT>[ \t\n]*                ;
<TEXT>"#"                       {BEGIN COMMENT;}
<COMMENT>\n                     {BEGIN TEXT;}
<COMMENT>[^\n]*         ;
%%

int yywrap() { return(-1); }
void sl_init_lex() { BEGIN TEXT; }
