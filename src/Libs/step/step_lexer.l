%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"
#include "step_bison_parser.h"
%}

%option warn
%option noyywrap
%option noyy_top_state
%option noinput
%option nounput
%option stack

%x unknow_pragma
%x omp_pragma omp_reduction
%x step_pragma
%x variable_list
%x schedule_arg

letter [a-zA-Z]
ident ({letter}|"_")({letter}|[0-9]|"_")*

%%
<INITIAL>. {yy_push_state(unknow_pragma); yymore();}
<unknow_pragma>.*\n {yy_pop_state(); step_lexerlval.string = strdup(yytext); return TK_RAW;}


<INITIAL>^(?i:omp) {yy_push_state(omp_pragma); return TK_OMP_PRAGMA;}
<omp_pragma>{
  \n {yy_pop_state(); return TK_EOL;}
  ("!"|"//")[^\n]* {step_lexerlval.string = strdup(yytext); return TK_COMMENT;}
  (?i:parallel) {step_lexerlval.integer = STEP_PARALLEL; return TK_OMP_PARALLEL;}
  (?i:do)|(?i:for) {step_lexerlval.integer = STEP_DO; return TK_OMP_LOOP;}
  (?i:master) {step_lexerlval.integer = STEP_MASTER; return TK_OMP_MASTER;}
  (?i:single) {step_lexerlval.integer = STEP_SINGLE; return TK_OMP_SINGLE;}
  (?i:barrier) {step_lexerlval.integer = STEP_BARRIER; return TK_OMP_BARRIER;}
  (?i:threadprivate) {/*is stored as both a directive and a clause */
    step_lexerlval.integer = STEP_THREADPRIVATE; yy_push_state(variable_list); return TK_OMP_THREADPRIVATE;}

  (?i:end) {step_lexerlval.integer = 0; return TK_OMP_END;}

  (?i:nowait) {return TK_OMP_NOWAIT;}
  (?i:shared) {yy_push_state(variable_list); return TK_OMP_SHARED;}
  (?i:private) {yy_push_state(variable_list); return TK_OMP_PRIVATE;}
  (?i:copyin) {yy_push_state(variable_list); return TK_OMP_COPYIN;}
  (?i:firstprivate) {yy_push_state(variable_list); return TK_OMP_FIRSTPRIVATE;}
  (?i:schedule) {yy_push_state(schedule_arg); return TK_OMP_SCHEDULE;}
  (?i:reduction) {yy_push_state(omp_reduction); return TK_OMP_REDUCTION;}
  (?i:default) {yy_push_state(variable_list); return TK_OMP_DEFAULT;}
}
<omp_reduction>{
  "(" {return TK_LPAREN;}
  "+" {step_lexerlval.integer = STEP_SUM_REDUCE; return TK_OPERATOR;}
  "*" {step_lexerlval.integer = STEP_PROD_REDUCE; return TK_OPERATOR;}
  (?i:max) {step_lexerlval.integer = STEP_MAX_REDUCE; return TK_OPERATOR;}
  (?i:min) {step_lexerlval.integer = STEP_MIN_REDUCE; return TK_OPERATOR;}
  ":" {yy_pop_state(); yy_push_state(variable_list); return TK_COLON;}
}


<INITIAL>^(?i:step) {yy_push_state(step_pragma); return TK_STEP_PRAGMA;}
<step_pragma>{
  \n {yy_pop_state(); return TK_EOL;}
  (?i:mpi) {step_lexerlval.integer = STEP_TRANSFORMATION_MPI; return TK_STEP_TRANSFORMATION;}
  (?i:no_mpi) {step_lexerlval.integer = STEP_TRANSFORMATION_OMP; return TK_STEP_TRANSFORMATION;}
  (?i:hybrid) {step_lexerlval.integer = STEP_TRANSFORMATION_HYBRID; return TK_STEP_TRANSFORMATION;}
  (?i:ignore) {step_lexerlval.integer = STEP_TRANSFORMATION_SEQ; return TK_STEP_TRANSFORMATION;}
}


<variable_list>{
"(" {return TK_LPAREN;}
"," {return TK_COMMA;}
")" {yy_pop_state(); return TK_RPAREN;}
{ident} {step_lexerlval.string = strdup(yytext); return TK_IDENT;}
}

<schedule_arg>{
"(" {return TK_LPAREN;}
"," {return TK_COMMA;}
")" {yy_pop_state(); return TK_RPAREN;}
{letter}+ {step_lexerlval.string = strdup(yytext); return TK_IDENT;}
[0-9]+ {step_lexerlval.string = strdup(yytext); return TK_IDENT;}
}

<*>[ \t] {}
<*>[^ \t] {pips_debug(1, "\tunexepected character: %s\n", yytext); return TK_ERROR;}

%%

