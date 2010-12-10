%{
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "p4a_include-OpenCL.h"

  // File Name
  char *file_name;

  // Pointer to the file descriptor
  extern FILE *yyin;

  //Line number
  int no_line;

  /** For types described with typedef

      Needing the true type for the sizeof function

      We save the equivalence between the typedef and the true type

      Stored in a list.
   */
  struct type_substitution * substitution_list=NULL;
  struct type_substitution * current_substitution;			     
%}

%union {
  char *string;
}
%token TYPEDEF PT_VIRG WRAPPER TAB PTR STRUCT
%token <string> NOM ARG TYPE

%type <string> type variable_name variable

%%

start : pattern start 
| pattern ;

/** We search for
    - typedef
    - the kernel_wrapper to analyse its argument list 
 */
pattern : type_def 
| wrapper 
| autre ;

autre : NOM      {/*printf("autre %d : %s\n",no_line,$1);*/}
| PT_VIRG ;

type_def : TYPEDEF type variable_name 
{
  //printf("Substitution de %s par %s\n",$2,$3);
  current_substitution = new_typedef($2,$3);
};

definition : NOM {
  //printf("definition %d : %s\n",no_line,$1);
};

wrapper : WRAPPER NOM args {
  //printf("kernel %d : %s\n",no_line,$2);
};

args : arg args
| arg ;

arg : type variable_name
{
  char * s = $1;
  if (strcmp($2,"cl_mem") == 0)
    s = "cl_mem";
  current_kernel->current = new_type(s);
};

type : TYPE {
  $$=$1;
  //printf("type %d : %s\n",no_line,$1);
}
| STRUCT TYPE {
  const char *str;
  asprintf(str,"struct %s",$2);
  $$=str;
  //printf("type %d : %s\n",no_line,str);
}
| TYPE PTR  {
  $$="cl_mem";
  //printf("type %d : %s *\n",no_line,$1);
}
| STRUCT TYPE PTR  {
  $$="cl_mem";
  //printf("type %d : struct %s *\n",no_line,$2);
};

/** variable name is decomposed between <<names>> and [ or ]
    symbols. Those latest are used to know if it is a pointer type.

    Two cases :
    - variable_name in the list argument : we only need the last value to
      know of it is a ]
    - in a typedef, only one variable_nema = the mlast value

    Thus, the $$ is designed to return only the last value of the list.

    Could be improved later on, but not sure that we need a more complex algo.
 */
variable_name : variable variable_name {
  $$ = $2;
  //printf("v1 : %s v2 : %s\n",$1,$2);
}
| variable 
{
  $$ = $1;
  //printf("v1 : %s\n",$1);
};

variable : ARG {
  $$ = $1; 
  //printf("argument %d : %s\n",no_line,$1);
 }
| TAB {
  $$ = "cl_mem";
};

%%
/** This function is automatically called when an error occurs.
 */

void yyerror(char *s)
{
  fprintf(stderr,"File %s, line %d : %s\n",file_name,no_line,s);
  exit(EXIT_FAILURE);
}

void parse_file(char *name)
{
  no_line = 1;
  file_name = name;
  if ((yyin = fopen(name,"r")) == NULL) {
    printf("File %s not available\n",name);
    exit(EXIT_FAILURE);
  }
  yyparse();
}

struct type_substitution * new_typedef(char *str1,char *str2)
{
  struct type_substitution * subs = (struct type_substitution *)malloc(sizeof(struct type_substitution));
  subs->definition = strdup(str1);
  subs->substitute = strdup(str2);
  subs->next = NULL;

  if (substitution_list == NULL)
    substitution_list = subs;
  else
    current_substitution->next = subs;
  return subs;
}

char * search_type_for_substitute(char *str)
{
  struct type_substitution * subs = substitution_list;
  while (subs) {
    if (strcmp(subs->substitute,str) == 0)
      return subs->definition;
    subs = subs->next;
  }
  return NULL;
}
