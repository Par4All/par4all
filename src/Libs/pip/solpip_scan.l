%{
/* Name     : solpip_scan.l
 * Package  : paf-util
 * Author   : F. Lamour, F. Dumontet
 * Date     : 25 march  93
 * Historic : 2 august 93, moved into (package) paf-util, AP
 * Documents:
 *
 * Comments :
 * Grammaire Lex pour interpreter une solution fournie par PIP en un "quast
 * newgen".
 */

#include "solpip_parse.h"
%}

               /* Pour que espace, cr, et tabulation soient ignores */
chiffre [0-9]
op [\/\-]
letter [A-Za-z_]
symboles {letter}({letter}|{chiffre})*

%%
"list"         {return (LST);}
"#"            {return (DIES);}
"["            {return (LCRO);}
"]"            {return (RCRO);}
"("            {return (LPAR);}
")"            {return (RPAR);}
"if"           {return (IF);}
"(newparm"     {return (NEWPARM);}
"div"          {return (DIV);}
"/"            {return (DIV_OP);}
"-"            {return (MOINS_OP);}

{chiffre}+     {sscanf(yytext,"%d",&yylval.valeur); return(ENTIER);}
{symboles}     {sscanf(yytext, "%c",&yylval.blabla); return(TEXTE);}
[ \t\n]       {;} 
%%

yywrap()
{return(1);}
