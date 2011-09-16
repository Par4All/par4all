%{
#include "defines-local.h"
#include <ctype.h> // for toupper

static void directive_block_begin(int type, string s);
static void directive_block_end(int type, string s);
static void directive_statement(int type, string s);
static void insert_optionnal_pragma(int type);
static step_directive get_current_step_directive(bool open);

static void add_clause_nowait(void);
static void add_clause_private(void);
static void add_clause_shared(void);
static void add_clause_reduction(void);

static statement current_statement = statement_undefined;
static pragma current_pragma = pragma_undefined;
static list current_list = list_undefined;
static int current_op = -1;
static int current_transform = -1;

static void set_step_transform(int type);
static void reset_step_transform(void);
static void init_current_list(void);
static void reset_current_list(void);
static void set_current_op(string op_name);
static void reset_current_op(void);

extern void step_comment2pragma_handle(statement stmt);

DEFINE_LOCAL_STACK(step_blocks, sequence);

%}

%option warn
%option noyywrap
%option noyy_top_state
%option stack

 /* DIRECTIVE step */
%x step
step (?i:step)

transform_mpi [ \t]*(?i:mpi)[ \t]*
transform_nompi  [ \t]*(?i:no_mpi)[ \t]*
transform_hybrid  [ \t]*(?i:hybrid)[ \t]*
transform_ignore  [ \t]*(?i:ignore)[ \t]*


 /* DIRECTIVE omp */
%x omp
omp (?i:omp)

%x parallel
parallel [ \t]*(?i:parallel)[ \t]*

%x loop
loop [ \t]*(?i:do)|(?i:for)[ \t]*

%x parallel_loop
parallel_loop {parallel}{loop}

%x end
end [ \t]*(?i:end)[ \t]*

%x barrier
barrier [ \t]*(?i:barrier)[ \t]*

%x master
master [ \t]*(?i:master)[ \t]*

%x single
single [ \t]*(?i:single)[ \t]*

 /* CLAUSE omp */

nowait [ \t]*(?i:nowait)[ \t]*

%x private
private [ \t]*(?i:private)[ \t]*

%x shared
shared [ \t]*(?i:shared)[ \t]*

%x reduction
reduction [ \t]*(?i:reduction)[ \t]*\([ \t]*
reduction_end [ \t]*\)[ \t]*
reduction_sep [ \t]*:[ \t]*
reduction_op ("+"|"*"|(?i:max)|(?i:min))



%x other
%x liste
liste_begin [ \t]*\([ \t]*
liste_end [ \t]*\)[ \t]*
liste_suite [ \t]*,[ \t]*
liste_element [^ \t,()\n]+

%%

<INITIAL>{
  {omp} {
    yy_push_state(omp);
    pips_debug(2,"begin state \"%s\" %d\n", yytext, YYSTATE);
  }
  {step} {
    yy_push_state(step);
    pips_debug(2,"begin state \"%s\" %d\n", yytext, YYSTATE);
  }
  . {
    yy_push_state(other);
    yymore();
    pips_debug(2,"begin state other : \"%s\" %d\n", yytext, YYSTATE);
  }
 }

 /**********************
      DIRECTIVES step
 **********************/
<step>{
  {transform_mpi} {
    pips_debug(2,"transform state %s %d\n", yytext, YYSTATE);
    set_step_transform(STEP_TRANSFORMATION_MPI);
  }
  {transform_nompi} {
    pips_debug(2,"transform state %s %d\n", yytext, YYSTATE);
    set_step_transform(STEP_TRANSFORMATION_OMP);
  }
  {transform_hybrid} {
    pips_debug(2,"transform state %s %d\n", yytext, YYSTATE);
    set_step_transform(STEP_TRANSFORMATION_HYBRID);
  }
  {transform_ignore} {
    pips_debug(2,"transform state %s %d\n", yytext, YYSTATE);
    set_step_transform(STEP_TRANSFORMATION_SEQ);
  }
  . {
    yymore();
    pips_debug(2,"step other : \"%s\" %d\n", yytext, YYSTATE);
  }
  \n {
    yy_pop_state();
    pips_debug(2,"end state step back to %d\n", YYSTATE);
  }
 }

 /**********************
      DIRECTIVES omp
 **********************/
<omp>{
  {parallel} {
    yy_push_state(parallel);
    pips_debug(2,"begin state %s %d\n", yytext, YYSTATE);
    directive_block_begin(STEP_PARALLEL, yytext);
  }
  {loop} {
    yy_push_state(loop);
    pips_debug(2,"begin state %s %d\n", yytext, YYSTATE);
    directive_block_begin(STEP_DO, yytext);
  }
  {parallel_loop} {
    yy_push_state(parallel_loop);
    pips_debug(2,"begin state %s %d\n", yytext, YYSTATE);
    directive_block_begin(STEP_PARALLEL_DO, yytext);
  }
  {barrier} {
    directive_statement(STEP_BARRIER, yytext);
  }
  {master} {
    yy_push_state(master);
    pips_debug(2,"begin state %s %d\n", yytext, YYSTATE);
    directive_block_begin(STEP_MASTER, yytext);
  }
  {single} {
    yy_push_state(single);
    pips_debug(2,"begin state %s %d\n", yytext, YYSTATE);
    directive_block_begin(STEP_SINGLE, yytext);
  }
  {end} {
    yy_push_state(end);
    pips_debug(2,"begin state \"%s\" %d\n", yytext, YYSTATE);
  }
  \n {
    yy_pop_state();
    pips_debug(2,"end state omp back to %d\n", YYSTATE);
  }
 }

<parallel>{
  . {
     yymore();
     pips_debug(2,"parallel remaining \"%s\"\n", yytext);
   }
   \n {
     yyunput('\n', step_omp_text);
     yy_pop_state();
     pips_debug(2,"end state parallel back to %d\n", YYSTATE);
   }
 }
<loop>{
  . {
     yymore();
     pips_debug(2,"loop remaining \"%s\"\n", yytext);
   }
  {nowait} {
    add_clause_nowait();
   }
  \n {
     yyunput('\n', step_omp_text);
     yy_pop_state();
     pips_debug(2,"end state loop back to %d\n", YYSTATE);
   }
 }
<parallel_loop>{
  . {
     yymore();
     pips_debug(2,"parallel_loop remaining \"%s\"\n", yytext);
   }
   \n {
     yyunput('\n', step_omp_text);
     yy_pop_state();
     pips_debug(2,"end state parallel_loop back to %d\n", YYSTATE);
   }
 }
<master>{
    . {
     yymore();
     pips_debug(2,"master remaining \"%s\"\n", yytext);
   }
   \n {
     yyunput('\n', step_omp_text);
     yy_pop_state();
     pips_debug(2,"end state master back to %d\n", YYSTATE);
   }
 }
<single>{
    . {
     yymore();
     pips_debug(2,"single remaining \"%s\"\n", yytext);
   }
   \n {
     yyunput('\n', step_omp_text);
     yy_pop_state();
     pips_debug(2,"end state single back to %d\n", YYSTATE);
   }
 }
<end>{
  {parallel}\n {
    directive_block_end(STEP_PARALLEL, yytext);
    pips_debug(2,"end state end +parallel back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {loop}\n {
    directive_block_end(STEP_DO, yytext);
    pips_debug(2,"end state end +loop back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {loop}{nowait}\n {
    directive_block_end(STEP_DO, yytext);
    add_clause_nowait();
    pips_debug(2,"end state end +loop back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {parallel_loop}\n {
    directive_block_end(STEP_PARALLEL_DO, yytext);
    pips_debug(2,"end state end +parallel_loop back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {master}\n {
    directive_block_end(STEP_MASTER, yytext);
    pips_debug(2,"end state end +master back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {single}\n {
    directive_block_end(STEP_SINGLE, yytext);
    pips_debug(2,"end state end +single back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  {single}{nowait}\n {
    directive_block_end(STEP_SINGLE, yytext);
    add_clause_nowait();
    pips_debug(2,"end state end +single back to %d\n", YYSTATE);
    yyunput('\n', step_omp_text);
    yy_pop_state();
  }
  .|\n {
    yyunput('\n', step_omp_text);
    yy_pop_state();
    pips_debug(2,"Error directive end\n");
  }
 }



 /****************
     CLAUSES omp
 ****************/

 /* private */
<parallel,loop,parallel_loop,single>{private} {
  yy_push_state(private);
 }
<private>{
  {liste_begin} {
    init_current_list();
    yy_push_state(liste);
    pips_debug(2,"private begin\n");
  }
  {liste_end} {
    add_clause_private();
    reset_current_list();
    pips_debug(2,"private end\n");
    yy_pop_state();
  }
  .|\n {
     pips_debug(2,"Error private\n");
  }
 }

 /* shared */
<parallel>{shared} {
  yy_push_state(shared);
 }
<shared>{
  {liste_begin} {
    init_current_list();
    yy_push_state(liste);
    pips_debug(2,"shared begin\n");
  }
  {liste_end} {
    add_clause_shared();
    reset_current_list();
    pips_debug(2,"shared end\n");
    yy_pop_state();
  }
  .|\n {
    pips_debug(2,"Error shared\n");
  }
 }

 /* reduction */
<parallel,loop,parallel_loop>{reduction} {
  yy_push_state(reduction);
  pips_debug(2,"reduction begin\n");
 }
<reduction>{
  {reduction_op} {
    pips_debug(2,"op : '%s'\n", yytext);
    set_current_op(yytext);
  }
  {reduction_sep} {
    init_current_list();
    yyunput('(', step_omp_text);
    yy_push_state(liste);
  }
  {reduction_end} {
    add_clause_reduction();
    reset_current_list();
    reset_current_op();
    pips_debug(2,"reduction end\n");
    yy_pop_state();
  }
  .|\n {
    pips_debug(2,"Error reduction\n");
  }
 }



 /* liste */
<liste>{
  {liste_begin} {
    pips_debug(4,"liste begin\n");
  }
  {liste_element} {
    pips_debug(4,"element \"%s\"", yytext);
    current_list = CONS(STRING, strdup(yytext), current_list);
  }
  {liste_suite} {
    pips_debug(4," suivant\n");
  }
  {liste_end} {
    yyunput(')', step_omp_text);
    yy_pop_state();
    pips_debug(4,"\nliste fin\n");
  }
  .|\n {
    pips_debug(4,"Erreur liste\n");
  }
 }

<other>.*\n {
  yy_pop_state();
  pips_debug(2,"end state other \"%s\" back to %d\n", yytext, YYSTATE);
  directive_statement(-1, yytext);
 }

%%

static void insert_optionnal_pragma(int type)
{
  statement new_stmt = make_plain_continue_statement();
  switch (type)
    {
    case STEP_DO:
      add_pragma_str_to_statement(new_stmt, "ompenddo\n", true);
      break;
    case STEP_PARALLEL_DO:
      add_pragma_str_to_statement(new_stmt, "ompendparalleldo\n", true);
      break;
    default:
      pips_user_error("Unexpected pragma type %d\n", type);
    }
  statement parent_stmt = (statement)gen_get_ancestor(statement_domain, current_statement);
  pips_assert("parent", parent_stmt != NULL);
  pips_assert("block", statement_block_p(parent_stmt));
  gen_insert_after(new_stmt, current_statement, statement_block(parent_stmt));
}

static void remove_old_pragma(void)
{
  pips_assert("statement defined", !statement_undefined_p(current_statement));
  pips_assert("pragma defined", !pragma_undefined_p(current_pragma));

  extensions es = statement_extensions(current_statement);

  list el = extensions_extension(es);

  extensions_extension(es) = NIL;
  FOREACH(EXTENSION, e, el)
    {
      pragma p=extension_pragma(e);
      if (p != current_pragma)
	extensions_extension(es) = gen_extension_cons(e, extensions_extension(es));
    }

  extensions_extension(es) = gen_nreverse(extensions_extension(es));
}

static void set_step_transform(int type)
{
  current_transform = type;
  remove_old_pragma();
  pips_debug(2, "step transform type : %d\n", current_transform);
}

static void reset_step_transform()
{
  const char* tranformation = get_string_property("STEP_DEFAULT_TRANSFORMATION");

  if (strncasecmp(tranformation, STEP_DEFAULT_TRANSFORMATION_MPI_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_MPI_TXT))==0)
    current_transform = STEP_TRANSFORMATION_MPI;
  else if (strncasecmp(tranformation, STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT))==0)
    current_transform = STEP_TRANSFORMATION_HYBRID;
  else if (strncasecmp(tranformation, STEP_DEFAULT_TRANSFORMATION_OMP_TXT, strlen(STEP_DEFAULT_TRANSFORMATION_OMP_TXT))==0)
    current_transform = STEP_TRANSFORMATION_OMP;
  else
    current_transform = STEP_TRANSFORMATION_SEQ;

  pips_debug(2, "step transform type : %d\n", current_transform);
}

static void add_step_directive(statement directive_stmt, int type, string s)
{
  remove_old_pragma();

  /* si le statement portant la directive n'est pas une sequence, on en fait une sequence */
  if (!statement_block_p(directive_stmt))
    {
      statement new_stmt = instruction_to_statement(statement_instruction(directive_stmt));
      move_statement_attributes(directive_stmt, new_stmt);

      statement_label(directive_stmt) = entity_empty_label();
      statement_comments(directive_stmt) = empty_comments;
      statement_declarations(directive_stmt) = NIL;
      statement_decls_text(directive_stmt) = NULL;
      statement_extensions(directive_stmt) = empty_extensions();

      statement_instruction(directive_stmt) = make_instruction_block(CONS(STATEMENT, new_stmt, NIL));
    }

  step_directive d = make_step_directive(type, directive_stmt, CONS(STEP_CLAUSE, make_step_clause_transformation(current_transform), NIL));

  sequence current_block = step_blocks_head();
  sequence_statements(current_block) = CONS(STATEMENT, directive_stmt, sequence_statements(current_block));

  step_directives_store(directive_stmt, d);
  add_pragma_str_to_statement(directive_stmt, strdup(concatenate(STEP_SENTINELLE, s, NULL)), false);
}

static void directive_block_begin(int type, string s)
{
  pips_debug(1,"DIRECTIVE block begin : %s\n",s);

  if(fortran_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  pips_assert("loop statement", statement_loop_p(current_statement)||statement_forloop_p(current_statement));
	  insert_optionnal_pragma(type);
	case STEP_PARALLEL:
	case STEP_MASTER:
	case STEP_SINGLE:
	case STEP_BARRIER:
	  {
	    statement directive_stmt = make_empty_block_statement();
	    add_step_directive(directive_stmt, type, s);
	    step_blocks_push(statement_sequence(directive_stmt));
	  }
	  break;
	default:
	  pips_user_error("unknow directive type %d\n", type);
	  break;
	}
    }
  else if (c_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  pips_assert("loop statement", statement_loop_p(current_statement)||statement_forloop_p(current_statement));
	  add_step_directive(current_statement, type, s);
	  break;
	case STEP_PARALLEL:
	case STEP_MASTER:
	case STEP_SINGLE:
	  pips_assert("block statement", statement_block_p(current_statement));
	  add_step_directive(current_statement, type, s);
	  break;
	case STEP_BARRIER:
	  add_step_directive(make_empty_block_statement(), type, s);
	  break;
	default:
	  pips_user_error("unknow directive type %d\n", type);
	  break;
	}
    }
  else
    {
      pips_user_error("language not supported");
    }
  reset_step_transform();
}

static void directive_block_end(int type, string s)
{
  /*
    -> substitution dans une sequence : I1,...,Ii+OMPpragma,... , Ij-1, Ij+OMPpragma, ..., In en :
    I1,..., Block(Ii, ..., Ij-1 )+STEPpragma, Ij..., In
  */
  if(fortran_module_p(get_current_module_entity()))
    {
      switch(type)
	{
	case STEP_DO:
	case STEP_PARALLEL_DO:
	  {
	    sequence current_block = step_blocks_head();
	    statement last_stmt = STATEMENT(CAR(sequence_statements(current_block)));

	    /* suppression du CONTINUE portant le pragma optionel */
	    while (empty_statement_or_labelless_continue_p(last_stmt))
	      {
		POP(sequence_statements(current_block));
		free_statement(last_stmt);
		last_stmt = STATEMENT(CAR(sequence_statements(current_block)));
	      }
	    pips_debug(2,"last statement %d\n", instruction_tag(statement_instruction(last_stmt)));

	    if(step_directives_bound_p(last_stmt))
	      {
		if (step_directive_type(step_directives_get(last_stmt)) != type)
		  pips_user_error("\nDirective end-loop not well formed\n");
		else
		  {
		    pips_debug(2,"loop directive already closed\n");
		    break;
		  }
	      }
	    else if (!statement_loop_p(last_stmt))
	      pips_user_error("\nDirective end-loop after a no-loop statement\n");
	  }
	default:
	  {
	    step_directive d = get_current_step_directive(true);
	    if (step_directive_undefined_p(d) || step_directive_type(d) != type)
	      pips_user_error("\nDirective not well formed : \"%s\"\n", s);

	    step_blocks_pop();
	    statement directive_stmt = step_directive_block(d);
	    sequence_statements(statement_sequence(directive_stmt)) = gen_nreverse(statement_block(directive_stmt));
	  }
	}
      remove_old_pragma();
    }
  pips_debug(1,"DIRECTIVE block end : %s\n",s);
}

static void directive_statement(int type, string s)
{
  pips_debug(1,"DIRECTIVE statement begin : %s\n",s);
  directive_block_begin(type, s);
  directive_block_end(type, s);
  pips_debug(1,"DIRECTIVE statement end : %s\n",s);
}


static void step_pragma_handle(statement stmt)
{
  pips_assert("statement undefined", statement_undefined_p(current_statement));
  current_statement = stmt;
  pips_assert("pragma undefined", pragma_undefined_p(current_pragma));

  reset_step_transform();

  FOREACH(EXTENSION, e, extensions_extension(statement_extensions(stmt)))
    {
      current_pragma = extension_pragma(e);
      if (pragma_string_p(current_pragma))
	{
	  string pragma_str = pragma_string(current_pragma);
	  pips_debug(1, "PRAGMA : \"%s\"\n",pragma_str);
	  STEP_DEBUG_STATEMENT(1, "on statement",current_statement);
	  pips_debug(1, "instruction %d\n", instruction_tag(statement_instruction(current_statement)));

	  if(pragma_str[strlen(pragma_str)-1]!= '\n')
	    pragma_str = strdup(concatenate(pragma_string(current_pragma),"\n",NULL));
	  else
	    pragma_str = strdup(pragma_string(current_pragma));

	  step_omp__scan_string(pragma_str);
	  step_omp_lex();

	  free(pragma_str);
	}
    }
  current_pragma = pragma_undefined;
  current_statement = statement_undefined;
}

static bool statement_filter(statement stmt)
{
  STEP_DEBUG_STATEMENT(2,"in",stmt);
  int size = step_blocks_size();
  int head_len = size?gen_length(sequence_statements(step_blocks_head())):0;
  pips_debug(2,"block_stack size :%d\thead_length=%d\n",size,head_len);

  /* Convertion des commentaires en pragma */
  if(fortran_module_p(get_current_module_entity()))
    step_comment2pragma_handle(stmt);

  /* Analyse et traitement des pragma */
  if (step_blocks_empty_p())
    {
      pips_debug(2,"Rien a faire au niveau du body du module\n");
    }
  else
    {
      statement parent_stmt = (statement)gen_get_ancestor(statement_domain, stmt);
      pips_assert("parent", parent_stmt != NULL);

      if (!ENDP(extensions_extension(statement_extensions(stmt))) &&
	  !statement_block_p(parent_stmt))
	{
	  /*
	    On transforme un statement portant un pragma en block s'il ne l'etait pas deja.
	    L'analyse aura lieu lors du parcours du block nouvellement cree
	  */
	  pips_debug(2,"Conversion en block\n");

	  statement new_stmt = instruction_to_statement(statement_instruction(stmt));
	  move_statement_attributes(stmt, new_stmt);

	  statement_label(stmt) = entity_empty_label();
	  statement_comments(stmt) = empty_comments;
	  statement_declarations(stmt) = NIL;
	  statement_decls_text(stmt) = NULL;
	  statement_extensions(stmt) = empty_extensions();

	  statement_instruction(stmt) = make_instruction_block(CONS(STATEMENT, new_stmt, NIL));
	}
      else
	{
	  pips_debug(2,"#### ANALYSE des PRAGMA ####\n");
	  step_pragma_handle(stmt);

	  /*
	    Si dans une sequence, on met a jour le block courrant
	    Si c'est une directive, elle a ete ajoute a son identification
	  */
	  if(!step_directives_bound_p(stmt) && statement_block_p(parent_stmt))
	    {
	      sequence current_block = step_blocks_head();
	      STEP_DEBUG_STATEMENT(2,"ajout au block courrant ",stmt);
	      sequence_statements(current_block)=CONS(STATEMENT, stmt,
						      sequence_statements(current_block));
	    }
	  else
	    {
	      pips_debug(2,"pas d'ajout au block courrant\n");
	    }
	}
    }

  STEP_DEBUG_STATEMENT(2,"out ",stmt);
  return true;
}


static void statement_rewrite(statement stmt)
{
  STEP_DEBUG_STATEMENT(3, "in", stmt);
  int size = step_blocks_size();
  int head_len = size?gen_length(sequence_statements(step_blocks_head())):0;
  pips_debug(4, "block_stack size :%d\thead_length=%d\n", size, head_len);

  STEP_DEBUG_STATEMENT(3, "out", stmt);
  return;
}

static bool sequence_filter(sequence __attribute__ ((unused)) seq)
{
  step_blocks_push(make_sequence(NIL));
  int size = step_blocks_size();
  int head_len = size?gen_length(sequence_statements(step_blocks_head())):0;
  pips_debug(4, "block_stack size :%d\thead_length=%d\n", size, head_len);
  return true;
}

static void sequence_rewrite(sequence seq)
{
  int size = step_blocks_size();
  sequence previous_block = step_blocks_nth(2);

  if(previous_block && !sequence_undefined_p(previous_block))
    {
      statement last_stmt = STATEMENT(CAR(sequence_statements(previous_block)));
      if (fortran_module_p(get_current_module_entity()) && step_directives_bound_p(last_stmt))
	pips_user_error("\nDirective not well formed\n");
    }

  sequence new_seq = step_blocks_pop();
  pips_debug(2, "block_stack size :%d\t new_seq_length=%d\n", size, (int)gen_length(sequence_statements(new_seq)));

  gen_free_list(sequence_statements(seq));
  sequence_statements(seq) = gen_nreverse(sequence_statements(new_seq));
  sequence_statements(new_seq) = NIL;
  free_sequence(new_seq);

  return;
}

void step_directive_parser(statement body)
{
  // Parse declaration txt
  step_comment2pragma_handle(statement_undefined);

  make_step_blocks_stack();

  // where the OpenMP constructs are identified
  gen_multi_recurse(body,
		    statement_domain, statement_filter, statement_rewrite,
		    sequence_domain, sequence_filter, sequence_rewrite,
		    NULL);

  free_step_blocks_stack();
}

/*
  Le statement conrespondant a la directive courante est en debut de sequence
  (l'ordre des statements dans les séquences est inversé dans la pile step_blocks)
*/
static step_directive get_current_step_directive(bool open)
{
  if(open && fortran_module_p(get_current_module_entity()))
    {
      pips_assert("stack size", step_blocks_size() > 1);

      sequence parent_block = step_blocks_nth(2);

      statement last_stmt = STATEMENT(CAR(sequence_statements(parent_block)));
      STEP_DEBUG_STATEMENT(1,"last_stmt 1",last_stmt);

      pips_assert("directive", step_directives_bound_p(last_stmt));
      pips_assert("sequence", statement_sequence_p(last_stmt));
      pips_assert("same sequence", statement_sequence(last_stmt)==step_blocks_head());

      return step_directives_get(last_stmt);
    }
  else if(!step_blocks_empty_p())
    {
      sequence current_block = step_blocks_head();

      if (!ENDP(sequence_statements(current_block)))
	{
	  statement last_stmt = STATEMENT(CAR(sequence_statements(current_block)));
	  STEP_DEBUG_STATEMENT(1,"last_stmt 2",last_stmt);

	  if(step_directives_bound_p(last_stmt))
	    return step_directives_get(last_stmt);
	}
    }

  return step_directive_undefined;
}

static void add_clause_nowait(void)
{
  step_directive d = get_current_step_directive(false);
  pips_assert("directive", !step_directive_undefined_p(d));
  step_directive_clauses(d)= CONS(STEP_CLAUSE, make_step_clause_nowait(), step_directive_clauses(d));
  pips_debug(1,"cause nowait ADDED\n");
}

static entity entity_from_user_name(string name)
{
  entity e = entity_undefined;

  if(fortran_module_p(get_current_module_entity()))
    {
      /* passage de name a NAME */
      size_t i;
      for (i=0; i<strlen(name); i++)
	name[i]=toupper(name[i]);
      string full_name = strdup(concatenate(entity_user_name(get_current_module_entity()), MODULE_SEP_STRING, name, NULL));
      e = gen_find_tabulated(full_name, entity_domain);
      if (entity_undefined_p(e))
	pips_user_error("\nEntity \"%s\" not found\n", full_name);
      free(full_name);
    }
  else if (c_module_p(get_current_module_entity()))
    {
      /* determiner le nom avec le bon scope... */
      statement stmt_declaration = current_statement;
      pips_debug(3,"##### ENTITY DECL #####\n");
      while (entity_undefined_p(e) && stmt_declaration)
	{
	  list decl = statement_declarations(stmt_declaration);
	  FOREACH(entity, ee, decl)
	    {
	      pips_debug(3, "entity decl : \"%s\"\n", entity_name(ee));
	      if (strcmp(name, entity_user_name(ee)) == 0)
		{
		  e = ee;
		  break;
		}
	    }

	  stmt_declaration = (statement)gen_get_ancestor(statement_domain, stmt_declaration);
	}
      if (entity_undefined_p(e))
	pips_user_error("\nEntity \"%s\" not found\n", name);
    }
  return e;
}

static void add_clause_private(void)
{
  step_directive d = get_current_step_directive(true);
  list le = NIL;

  FOREACH(STRING, name, current_list)
    {
      entity e = entity_from_user_name(name);

      pips_assert("entity", !entity_undefined_p(e));
      le = CONS(ENTITY, e, le);

      pips_debug(2,"add private variable : %s (entity_name : %s)\n", name, entity_name(e));
      free(name);
    }
  gen_free_list(current_list);
  step_directive_clauses(d)= CONS(STEP_CLAUSE, make_step_clause_private(le), step_directive_clauses(d));
  pips_debug(1,"cause private ADDED\n");
}

static void add_clause_shared(void)
{
  step_directive d = get_current_step_directive(true);
  list le = NIL;

  FOREACH(STRING, name, current_list)
    {
      entity e = entity_from_user_name(name);

      pips_assert("entity", !entity_undefined_p(e));
      le = CONS(ENTITY, e, le);

      pips_debug(2,"add shared variable : %s (entity_name : %s)\n", name, entity_name(e));
      free(name);
    }
  gen_free_list(current_list);
  step_directive_clauses(d)= CONS(STEP_CLAUSE, make_step_clause_shared(le), step_directive_clauses(d));
  pips_debug(1,"cause shared ADDED\n");
}

static void init_current_list(void)
{
  pips_assert("list undefined", list_undefined_p(current_list));
  current_list = NIL;
}

static void reset_current_list(void)
{
  current_list = list_undefined;
}

static void set_current_op(string op)
{
  assert(current_op == -1);
  if (!strcasecmp(op,"+")) current_op = STEP_SUM;
  else if (!strcasecmp(op,"*")) current_op = STEP_PROD;
  else if (!strcasecmp(op,"min")) current_op = STEP_MIN;
  else if (!strcasecmp(op,"max")) current_op = STEP_MAX;
  else
    pips_user_error("reduction operator : \"%s\" not yet implemented\n\n",op);
}

static void reset_current_op()
{
  assert(current_op != -1);
  current_op = -1;
}

static void add_clause_reduction(void)
{
  step_directive d = get_current_step_directive(true);
  pips_assert("some variable", !ENDP(current_list));

  map_entity_int reductions = make_map_entity_int();

  FOREACH(STRING, name, current_list)
    {
      entity e = entity_from_user_name(name);

      pips_assert("entity", !entity_undefined_p(e));
      pips_assert("first reduction", !bound_map_entity_int_p(reductions,e));

      extend_map_entity_int(reductions, e, current_op);
      pips_debug(2,"add reduction %d variable : %s (entity_name : %s)\n", current_op, name, entity_name(e));
      free(name);
    }

  gen_free_list(current_list);
  step_directive_clauses(d)= CONS(STEP_CLAUSE, make_step_clause_reduction(reductions), step_directive_clauses(d));
  pips_debug(1,"cause reduction ADDED\n");
}
