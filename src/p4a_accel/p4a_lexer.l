/** @file lexique.l
* @brief Définition des mots du lexique
*/

%x comment define

%{
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "y.tab.h"
  
  extern int no_line;

%}

%a 10000
%p 10000
%o 15000
%e 5000
%n 2000

chiffre	[0-9]
exposant [DdEe][-+]?{chiffre}+
alnum [0-9A-Za-z]
comment_text=([^*]"/"[^*]|[^/]"*"[^/]|"*"[^/]|"/"[^*])*

%%
\/\/[^\n]*\n 	  {no_line++;}	// Skip line comment
\n	          {no_line++;}
[ \t]+ ;			// Skip blank and tabulation
"/*"         {BEGIN(comment);}
"typedef"                  return TYPEDEF; 
"#define"     {BEGIN(define);} 
";"                        return PT_VIRG; 
"P4A_accel_kernel_wrapper" return WRAPPER;

<comment>"*/" {BEGIN(INITIAL);}
<comment>\n        {no_line++;}
<comment>([^*/\n])*          |
<comment>([^*]"/"[^*])*      |
<comment>([^/]"*"[^/])*      |
<comment>("*"[^/])*          |
<comment>("/"[^*])*          ;

<define>\n         {no_line++;BEGIN(INITIAL);}
<define>"\\n"      {no_line++;}
<define>([^\n])*             ;

[^$ \t\n:;,\"\[\]=]+ {
  yylval.string = (char *)strdup(yytext);
  return NOM;
}

%%
/** @fn yywrap()
* @brief Procédure de fermeture du fichier lu.
*/

int yywrap()
{
   fclose(yyin);
   yy_delete_buffer(YY_CURRENT_BUFFER);
   return 1;
}

