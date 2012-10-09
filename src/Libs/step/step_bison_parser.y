%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"
#include "step_lexer.h"


extern void set_current_transform(int transform);

extern entity entity_from_user_name(string name);
extern void remove_old_pragma(void);
extern step_directive begin_omp_construct(int type, string s);
extern step_directive end_omp_construct(int type);

static step_directive concat_drt_clauses(step_directive drt, cons *liste_clauses);
static step_clause step_handle_reduction_clause(cons *ident_liste, int op);
static cons * step_add_to_ident_list(string name, cons * list);

static string pragma_str_original=NULL;

void step_error(const char *s)
{
  pips_user_error("\nParsing :\n%s\n%s at '%s'\n", pragma_str_original, s, step_text);
}

%}

/* Bison declarations */
%union {
  char* string;
  cons * liste;
  int integer;
  step_directive step_directive;
  step_clause step_clause;
}
%debug

%token TK_EOL TK_ERROR
%token TK_LPAREN TK_RPAREN
%token TK_COLON TK_COMMA

%token <string> TK_IDENT TK_COMMENT
%type <liste> ident_list string_list

%token <integer> TK_OPERATOR

%token <string> TK_RAW


%token TK_OMP_PRAGMA
%token <integer> TK_OMP_PARALLEL TK_OMP_LOOP TK_OMP_END TK_OMP_BARRIER TK_OMP_MASTER TK_OMP_SINGLE TK_OMP_THREADPRIVATE
%token TK_OMP_SHARED TK_OMP_PRIVATE TK_OMP_NOWAIT TK_OMP_REDUCTION TK_OMP_DEFAULT TK_OMP_COPYIN TK_OMP_FIRSTPRIVATE TK_OMP_SCHEDULE

%type <step_directive> omp_directive
%type <liste> omp_parallel_clauses omp_loop_clauses omp_end_loop_clauses omp_parallel_loop_clauses
%type <step_clause> omp_shared omp_private omp_reduction omp_copyin omp_firstprivate omp_schedule
%type <liste> omp_threadprivate_listvar

%token TK_STEP_PRAGMA
%token <integer> TK_STEP_TRANSFORMATION

%%

pragma:
  TK_RAW { pips_user_warning("unknown pragma : %s\n", $1);}

| TK_OMP_PRAGMA
    { pips_debug(1,"OMP pragma begin\n"); remove_old_pragma(); }
  omp_directive omp_comment TK_EOL
    { pips_debug(1,"OMP pragma end\n"); step_directive_print($3); }

| TK_STEP_PRAGMA
    { pips_debug(1,"STEP pragma begin\n"); remove_old_pragma(); }
  step_transformation TK_EOL
    { pips_debug(1,"STEP pragma end\n"); }
;

step_transformation:
  TK_STEP_TRANSFORMATION { pips_debug(1, "STEP transform : %d\n", $1); set_current_transform($1);}
;

omp_comment:
| TK_COMMENT { pips_debug(1,"COMMENT : %s\n", $1);}
;

omp_directive:
   TK_OMP_PARALLEL
      { $<step_directive>$ = begin_omp_construct($1, "PARALLEL"); }
   omp_parallel_clauses
      { $$ = concat_drt_clauses($<step_directive>2, $3); }
|  TK_OMP_LOOP
      { $<step_directive>$ = begin_omp_construct($1, "LOOP"); }
   omp_loop_clauses
      { $$ = concat_drt_clauses($<step_directive>2, $3); }
| TK_OMP_PARALLEL TK_OMP_LOOP
      { $<step_directive>$=begin_omp_construct(STEP_PARALLEL_DO, "PARALLEL LOOP");}
  omp_parallel_loop_clauses
      { $$ = concat_drt_clauses($<step_directive>3, $4); }
| TK_OMP_MASTER
      { $$ = begin_omp_construct($1, "MASTER");}
| TK_OMP_SINGLE
      { $$ = begin_omp_construct($1, "SINGLE");}
| TK_OMP_BARRIER
      { step_directive drt = begin_omp_construct($1, "BARRIER");
	end_omp_construct($1);
	$$ = drt; }
| TK_OMP_THREADPRIVATE
      { $<step_directive>$ = begin_omp_construct($1, "THREADPRIVATE");}
  omp_threadprivate_listvar
      { $$ = concat_drt_clauses($<step_directive>2, $3);
	end_omp_construct($1);}
| TK_OMP_END TK_OMP_PARALLEL
      { $$ = end_omp_construct($2);}
| TK_OMP_END TK_OMP_LOOP omp_end_loop_clauses
      { $$ = concat_drt_clauses(end_omp_construct($2), $3); }
| TK_OMP_END TK_OMP_PARALLEL TK_OMP_LOOP
      { $$ = end_omp_construct(STEP_PARALLEL_DO);}
| TK_OMP_END TK_OMP_MASTER
      { $$ = end_omp_construct($2);}
| TK_OMP_END TK_OMP_SINGLE
      { $$ = end_omp_construct($2);}
;

omp_parallel_clauses: {$$=NIL;}
| omp_parallel_clauses omp_shared
    { pips_debug(1, "clause SHARED\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_clauses omp_private
    { pips_debug(1, "clause PRIVATE\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_clauses omp_copyin
    { pips_debug(1, "clause COPYIN\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_clauses omp_firstprivate
    { pips_debug(1, "clause FIRSTPRIVATE\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_clauses omp_reduction
    { pips_debug(1, "clause REDUCTION\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_clauses omp_default
    { pips_debug(1, "clause DEFAULT SKIPPED\n");
      $$ = $1; }
;


omp_loop_clauses: {$$=NIL;}
| omp_loop_clauses omp_private
   { pips_debug(1, "clause PRIVATE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1);}
| omp_loop_clauses omp_firstprivate
   { pips_debug(1, "clause FIRSTPRIVATE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1);}
| omp_loop_clauses omp_reduction
   { pips_debug(1, "clause REDUCTION\n");
     $$ = CONS(STEP_CLAUSE, $2, $1);}
| omp_loop_clauses omp_schedule
   { pips_debug(1, "clause SCHEDULE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1);}
| omp_loop_clauses TK_OMP_NOWAIT
   { pips_debug(1, "clause NOWAIT\n");
     $$ = CONS(STEP_CLAUSE, make_step_clause_nowait(), $1);}
;

omp_end_loop_clauses:{$$=NIL;}
| omp_loop_clauses TK_OMP_NOWAIT {pips_debug(1, "clause NOWAIT\n");
    $$ = CONS(STEP_CLAUSE, make_step_clause_nowait(), $1);}
;

omp_parallel_loop_clauses: { $$=NIL; }
| omp_parallel_loop_clauses omp_shared
   { pips_debug(1, "clause SHARED\n");
     $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_private
   { pips_debug(1, "clause PRIVATE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_copyin
    { pips_debug(1, "clause COPYIN\n");
      $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_firstprivate
   { pips_debug(1, "clause FIRSTPRIVATE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_reduction
   { pips_debug(1, "clause REDUCTION\n");
     $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_schedule
   { pips_debug(1, "clause SCHEDULE\n");
     $$ = CONS(STEP_CLAUSE, $2, $1); }
| omp_parallel_loop_clauses omp_default
   { pips_debug(1, "clause DEFAULT SKIPPED\n");
     $$ = $1; }
;

omp_shared: TK_OMP_SHARED {pips_debug(2, "SHARED begin \n");}
TK_LPAREN ident_list TK_RPAREN {pips_debug(2, "SHARED end\n");
    $$ = make_step_clause_shared($4);}
;

omp_private:
  TK_OMP_PRIVATE
    { pips_debug(2, "PRIVATE begin \n");}
  TK_LPAREN ident_list TK_RPAREN
    { pips_debug(2, "PRIVATE end\n");
      $$ = make_step_clause_private($4);}
;

omp_copyin:
  TK_OMP_COPYIN
    { pips_debug(2, "COPYIN begin \n");}
  TK_LPAREN ident_list TK_RPAREN
    { pips_debug(2, "COPYIN end\n");
      $$ = make_step_clause_copyin($4);}
;

omp_threadprivate_listvar :
     { pips_debug(2, "THREADPRIVATE begin \n");}
   TK_LPAREN ident_list TK_RPAREN
     { pips_debug(2, "THREADPRIVATE end\n");
       $$ = CONS(STEP_CLAUSE, make_step_clause_threadprivate($3),NIL);}
;

omp_firstprivate:
  TK_OMP_FIRSTPRIVATE
    { pips_debug(2, "FIRSTPRIVATE begin \n");}
  TK_LPAREN ident_list TK_RPAREN
    { pips_debug(2, "FIRSTPRIVATE end\n");
      $$ = make_step_clause_firstprivate($4);}
;

omp_schedule:
  TK_OMP_SCHEDULE
    { pips_debug(2, "SCHEDULE begin \n");}
  TK_LPAREN string_list TK_RPAREN
    { pips_debug(2, "SCHEDULE end\n");
      $$ = make_step_clause_schedule(gen_nreverse($4)); }
;

omp_reduction:
  TK_OMP_REDUCTION
     { pips_debug(2, "REDUCTION begin\n"); }
  TK_LPAREN TK_OPERATOR
     { pips_debug(2, "OPERATOR %d\n", $4); }
  TK_COLON ident_list TK_RPAREN
     { pips_debug(2, "REDUCTION end\n");
       $$ = step_handle_reduction_clause($7, $4); }
;

omp_default:
  TK_OMP_DEFAULT
    { pips_debug(2, "DEFAULT begin\n"); }
  TK_LPAREN TK_IDENT  TK_RPAREN
    { pips_debug(2, "DEFAULT %s end\n", $4); }
;

string_list:
TK_IDENT
    { pips_debug(2, "first string %s\n", $1);
      $$ = CONS(STRING, $1, NIL);
    }
| string_list TK_COMMA TK_IDENT
    { pips_debug(2, "next IDENT %s\n", $3);
      $$ = CONS(STRING, $3, $1);
    }
;

ident_list:
  TK_IDENT
    { pips_debug(2, "first IDENT %s\n", $1);
      $$ = step_add_to_ident_list($1, NIL);
    }
| ident_list TK_COMMA TK_IDENT
    { pips_debug(2, "next IDENT %s\n", $3);
      $$ = step_add_to_ident_list($3, $1);
    }
;

%%

static cons * step_add_to_ident_list(string name, cons * list)
{
  pips_debug(2, "begin\n");
  entity e = entity_from_user_name(name);
  pips_assert("entity", !entity_undefined_p(e));
  pips_debug(2, "end\n");
  return CONS(ENTITY, e, list);
}

static step_clause step_handle_reduction_clause(cons *ident_liste, int op)
{
  pips_debug(2, "begin\n");
  map_entity_int reductions = make_map_entity_int();
  FOREACH(ENTITY, e, ident_liste)
    {
      pips_assert("first reduction", !bound_map_entity_int_p(reductions, e));
      pips_debug(2,"add reduction %d variable : %s)\n", op, entity_name(e));
      extend_map_entity_int(reductions, e, op);
    }
  gen_free_list(ident_liste);
  pips_debug(2, "end\n");
  return make_step_clause_reduction(reductions);
}

static step_directive concat_drt_clauses(step_directive drt, cons *liste_clauses)
{
  pips_debug(2, "begin\n");
  step_directive_clauses(drt)= gen_nconc(liste_clauses, step_directive_clauses(drt));
  pips_debug(2, "end\n");
  return drt;
}

void step_bison_parse(pragma pgm, statement stmt)
{
  string pragma_str = pragma_string(pgm);

  if(pragma_str[strlen(pragma_str)-1]!= '\n')
    pragma_str = strdup(concatenate(pragma_string(pgm),"\n",NULL));
  else
    pragma_str = strdup(pragma_string(pgm));

  pips_debug(1, "############### PARSER BEGIN ################\nwith :\n%s\non stmt : %p\n", pragma_str, stmt);
  ifdebug(4)
    yydebug=1;
  pragma_str_original = strdup(pragma_str);
  step__scan_string(pragma_str);
  step_parse();
  free(pragma_str_original);
  pips_debug(1, "############### PARSER END ################\n");

  free(pragma_str);
}
