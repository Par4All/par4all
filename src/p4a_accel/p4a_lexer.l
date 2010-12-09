/** @file lexique.l
* @brief Définition des mots du lexique
*/

%x comment define wrapper argument type

%{
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "y.tab.h"
#include "p4a_include-OpenCL.h"

  
  extern int no_line;

%}

%a 10000
%p 10000
%o 15000
%e 5000
%n 2000

%%
\/\/[^\n]*\n 	           {no_line++;}	// Skip line comment
\n	                   {no_line++;}
[ \t]*                     ; // Skip blank and tabulation
"/*"                       {BEGIN(comment);}
"typedef"                  {BEGIN(type);return TYPEDEF;} 
"#define"                  {BEGIN(define);} 
";"                        return PT_VIRG; 
"P4A_accel_kernel_wrapper" {BEGIN(wrapper);return WRAPPER;}

<comment>"*/"              {BEGIN(INITIAL);}
<comment>\n                {no_line++;}
<comment>([^*/\n])*          |
<comment>([^*]"/"[^*])*      |
<comment>([^/]"*"[^/])*      |
<comment>("*"[^/])*          |
<comment>("/"[^*])*          ;

<define>\n                   {no_line++;BEGIN(INITIAL);}
<define>"\\n"                {no_line++;}
<define>([^\n])*             ;

<wrapper>"("                 {BEGIN(type);}
<wrapper>\n                  {no_line++;}
<wrapper>[ \t]*               ;
<wrapper>[^ \t\n)(]+ {
  yylval.string = (char *)strdup(yytext);
  return NOM;
}

<type>"struct"                      {return STRUCT;}
<type>"P4A_accel_global_address"                  ;
<type>"P4A_accel_constant_address"                ;
<type>"*"                              {return PTR;}               
<type>[ \t]*                                      ;
<type>[^ \t\n)(,*\[\]]+ {
  BEGIN(argument);
  yylval.string = (char *)strdup(yytext);
  return TYPE;
}


<argument>")" {current_kernel->n_args++;BEGIN(INITIAL);} //end of argument list
<argument>";"                         {BEGIN(INITIAL);} //end of typedef
<argument>","   {BEGIN(type);current_kernel->n_args++;} //separate arg in a list
<argument>"["                              {return TAB;}
<argument>"]"                              {return TAB;}               
<argument>\n                                {no_line++;}
<argument>[ \t]*                                      ;
<argument>[^ \t\n)(,*\[\];]+ {
  yylval.string = (char *)strdup(yytext);
  return ARG;
}

[^$ \t\n;]+ {
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
   yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE));
   return 1;
}

