/* Copyright 2007, 2008, 2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h" // for STEP_DEBUG_STATEMENT
#include "control.h" // for module_reorder

extern void step_directive_parser(statement body);

GENERIC_LOCAL_FUNCTION(directives, step_directives);

void step_directives_init()
{
  const char *module_name = entity_user_name(get_current_module_entity());
  set_directives((step_directives)db_get_memory_resource(DBR_STEP_DIRECTIVES, module_name, true));
}

void step_directives_reset()
{
  reset_directives();
}

static void step_directives_save()
{
  const char *module_name = entity_user_name(get_current_module_entity());
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_DIRECTIVES, module_name, get_directives());
  reset_directives();
}


step_directive step_directives_get(statement stmt)
{
  return load_directives(stmt);
}

bool step_directives_bound_p(statement stmt)
{
  return bound_directives_p(stmt);
}

void step_directives_store(statement stmt, step_directive d)
{
  assert(statement_block_p(stmt));
  store_directives(stmt, d);
}

#define SB_LIST_VARIABLE(sb, list_var, txt_begin) 			\
  if(!set_empty_p(list_var))						\
    {									\
      string s = string_undefined;					\
      FOREACH(ENTITY, variable, set_to_sorted_list(list_var, (gen_cmp_func_t)compare_entities)) \
	{								\
	  if(s == string_undefined)					\
	    s=strdup(concatenate(txt_begin, entity_user_name(variable), NULL)); \
	  else								\
	    s=strdup(concatenate(", ", entity_user_name(variable), NULL)); \
	  string_buffer_append(sb, s);					\
	}								\
      string_buffer_append(sb, strdup(")"));				\
    }

bool step_directive_to_strings(step_directive d, bool is_fortran, string *begin_txt, string *end_txt)
{
  bool block_directive = true;
  bool end_directive = is_fortran;

  string directive_txt;
  switch(step_directive_type(d))
    {
    case STEP_PARALLEL:
      directive_txt = strdup("parallel");
      break;
    case STEP_DO:
      directive_txt = strdup(is_fortran?"do":"for");
      block_directive = is_fortran;
      break;
    case STEP_PARALLEL_DO:
      directive_txt = strdup(is_fortran?"parallel do":"parallel for");
      block_directive = is_fortran;
      break;
    case STEP_MASTER:
      directive_txt = strdup("master");
      break;
    case STEP_SINGLE:
      directive_txt = strdup("single");
      break;
    case STEP_BARRIER:
      directive_txt = strdup("barrier");
      block_directive = false;
      end_directive = false;
      break;
    default: assert(0);
    }

  /*  clause */
  set private_l = set_make(set_pointer);
  set shared_l = set_make(set_pointer);
  bool nowait = false;

  int op;
  set reductions_l[STEP_OP_UNDEFINED];
  for(op=0; op<STEP_OP_UNDEFINED; op++)
    reductions_l[op] = set_make(set_pointer);

  FOREACH(STEP_CLAUSE, c, step_directive_clauses(d))
    {
      switch (step_clause_tag(c))
	{
	case is_step_clause_private:
	  set_append_list(private_l, step_clause_private(c));
	  break;
	case is_step_clause_shared:
	  set_append_list(shared_l, step_clause_shared(c));
	  break;
	case is_step_clause_nowait:
	  nowait = true;
	  break;
	case is_step_clause_reduction:
	  MAP_ENTITY_INT_MAP(variable, op, {
	      set_add_element(reductions_l[op], reductions_l[op], variable);
	    }, step_clause_reduction(c));
	  break;
	case is_step_clause_transformation:
	  break;
	default: assert(0);
	}
    }


  if(end_directive)
    *end_txt = strdup(concatenate("omp end ",directive_txt, nowait?" nowait":"", NULL));
  else
    *end_txt = string_undefined;

  string_buffer sb = string_buffer_make(false);
  string_buffer_cat(sb, strdup("omp "), strdup(directive_txt), NULL);

  SB_LIST_VARIABLE(sb, private_l, " private(");
  SB_LIST_VARIABLE(sb, shared_l, " shared(");

  string op_txt[STEP_OP_UNDEFINED]={" reduction(*: "," reduction(max: "," reduction(min: "," reduction(+: "};
  for(op=0; op<STEP_OP_UNDEFINED; op++)
    SB_LIST_VARIABLE(sb, reductions_l[op], op_txt[op]);

  if(nowait && !end_directive)
    string_buffer_append(sb, strdup(" nowait"));

  *begin_txt = string_buffer_to_string(sb);
  string_buffer_free_all(&sb);

  ifdebug(4)
    {
      printf("øøøøøøøøø directive begin : %s\n", *begin_txt);
      printf("øøøøøøøøø directive end : %s\n", end_directive?*end_txt:"");
    }

  return block_directive;
}


statement step_directive_basic_workchunk(step_directive d)
{
  statement stmt = step_directive_block(d);

  switch(step_directive_type(d))
    {
    case STEP_DO:
    case STEP_PARALLEL_DO:
      {
	// on retourne le corps de boucle
	list block = statement_block(stmt);
	pips_assert("1 statement", gen_length(block) == 1);

	stmt = STATEMENT(CAR(block));

	if(statement_loop_p(stmt))
	  stmt = loop_body(statement_loop(stmt));
	else if (statement_forloop_p(stmt))
	  stmt = forloop_body(statement_forloop(stmt));
	else
	  pips_assert("not a loop", false);
	break;
      }
    default:
      {
	// on retourne le block de la directive
      }
    }

  return stmt;
}

list step_directive_basic_workchunk_index(step_directive d)
{
  list index_l = NIL;
  statement stmt = step_directive_block(d);

  switch(step_directive_type(d))
    {
    case STEP_DO:
    case STEP_PARALLEL_DO:
      {
	list block = statement_block(stmt);
	pips_assert("1 statement", gen_length(block) == 1);

	stmt = STATEMENT(CAR(block));

	if(statement_loop_p(stmt))
	  index_l = CONS(ENTITY, loop_index(statement_loop(stmt)), index_l);
	else if (statement_forloop_p(stmt))
	  {
	    expression init = forloop_initialization(statement_forloop(stmt));
	    pips_assert("an assignment", assignment_expression_p(init));
	    list assign_params = call_arguments(syntax_call(expression_syntax(init)));
	    expression lhs = EXPRESSION(CAR(assign_params));
	    entity e = expression_to_entity(lhs);
	    pips_assert("an entity", !entity_undefined_p(e));
	    index_l = CONS(ENTITY, e, index_l);
	  }
	else
	  pips_assert("not a loop", false);
	break;
      }
    default:
      {
	// on retourne la liste vide
      }
    }

  return gen_nreverse(index_l);
}

void step_directive_print(step_directive d)
{
  int type = step_directive_type(d);
  list clauses = step_directive_clauses(d);
  string begin_txt, end_txt;
  bool is_fortran = fortran_module_p(get_current_module_entity());
  bool is_block_construct = step_directive_to_strings(d, is_fortran, &begin_txt, &end_txt);

  pips_debug(1, "====> TYPE %d : \nNB clauses : %d\n\tdirective begin : %s\n",
	     type, (int)gen_length(clauses), begin_txt);
  if (is_block_construct && !empty_comments_p(end_txt)) pips_debug(1,"\tdirective end : %s\n", end_txt);

  ifdebug(1)
    {
      statement stmt = step_directive_block(d);
      assert(!statement_undefined_p(stmt));
      pips_debug(1, "----> on statement :\n");
      print_statement(stmt);
      pips_debug(1, "\n");
    }
  ifdebug(2)
    {
      statement stmt = step_directive_basic_workchunk(d);
      assert(!statement_undefined_p(stmt));

      string index_str=strdup("");
      FOREACH(ENTITY, e , step_directive_basic_workchunk_index(d))
	{
	  string previous = index_str;
	  index_str = strdup(concatenate(previous, entity_local_name(e), " ", NULL));
	  free(previous);
	}
      pips_debug(2, "\n----> basic workchunk (index : [%s] )\n", index_str);
      print_statement(stmt);
      pips_debug(2, "\n");
    }
}

bool step_parser(const char* module_name)
{
  debug_on("STEP_PARSER_DEBUG_LEVEL");
  pips_debug(1, "%d module_name = %s\n", __LINE__, module_name);

  statement stmt = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_current_module_entity(local_name_to_top_level_entity(module_name));

  init_directives();

  step_directive_parser(stmt);

  ifdebug(1)
    {
      STEP_DIRECTIVES_MAP(block_stmt, d,
			  {
			    assert(!statement_undefined_p(block_stmt));
			    step_directive_print(d);
			  }, get_directives());
    }

  step_directives_save();
  reset_current_module_entity();


  module_reorder(stmt);
  if(ordering_to_statement_initialized_p())
    reset_ordering_to_statement();

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stmt);

  pips_debug(1, "Fin step_directives\n");
  debug_off();

  return true;
}
