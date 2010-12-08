%{
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

  int parse_file(char *name);

  // File Name
  char *file_name;

  // Pointer to the file descriptor
  extern FILE *yyin;

  //Line number
  int no_line;
%}

%union {
  double real;
  int integer;
  char *string;
}
%token TYPEDEF PT_VIRG WRAPPER PAR_L PAR_R
%token <string> NOM

%%

start : pattern start 
| pattern ;

pattern : type_def 
| wrapper 
| autre ;

autre : NOM      {printf("autre %d : %s\n",no_line,$1);}
| PT_VIRG ;

type_def : TYPEDEF definitions PT_VIRG ;

definitions : definition definitions
| definition ;

definition : NOM      {printf("definition %d : %s\n",no_line,$1);};

wrapper : WRAPPER ;

/*
args : arg args
| arg ;

arg : NOM      {printf("arg : %s\n",$1);};
*/

%%
/** This function is automatically called when an error occurs.
 */

void yyerror(char *s)
{
  fprintf(stderr,"File %s, line %d : %s\n",file_name,no_line,s);
  exit(EXIT_FAILURE);
}

int parse_file(char *name)
{
  no_line = 1;
  file_name = name;
  yyin = fopen(name,"r");
  yyparse();
  return EXIT_SUCCESS;
}
