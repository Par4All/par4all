/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

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
