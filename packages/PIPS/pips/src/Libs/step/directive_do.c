/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"

#define RANGENAME_LOW_SUFFIX "_L"
#define RANGENAME_UP_SUFFIX "_U"

static entity clone_scalar(string name,entity module,entity scalar)
{
  entity e;

  pips_debug(1,"name = %s, module = %p, scalar = %p\n", name, module, scalar);

  pips_assert("scalar",entity_variable_p(scalar));
  pips_assert("dim",ENDP(variable_dimensions(type_variable(entity_type(scalar)))));
  e = make_scalar_entity(name, entity_user_name(module),
			    copy_basic(variable_basic(type_variable(entity_type(scalar)))));

  pips_debug(1,"e = %p\n", e);
  return e;
}

static string step_find_loop_range_suffix_id(string name)
{ 

  int id;
  string id_s, lower, upper;
  
  pips_debug(1,"name = %s\n", name);

  id_s = strdup("");
  lower = strdup(concatenate(name, RANGENAME_LOW_SUFFIX, NULL));
  upper = strdup(concatenate(name, RANGENAME_UP_SUFFIX, NULL));

  id = 1;
  while (!entity_undefined_p(gen_find_tabulated(lower, entity_domain)) ||
	 !entity_undefined_p(gen_find_tabulated(upper, entity_domain)))
    {
      free(id_s);id_s=i2a(id++);
      free(lower);lower=strdup(concatenate(name, RANGENAME_LOW_SUFFIX, id_s, NULL));
      free(upper);upper=strdup(concatenate(name, RANGENAME_UP_SUFFIX, id_s, NULL));
    } 

  pips_debug(1,"id_s = %s\n", id_s);

  return id_s;
}

static void clause_handling(entity directive_module, directive d)
{
  pips_debug(1,"d = %p", d);
  // reduction handling
  directive_clauses(d) = gen_nconc(directive_clauses(d),
				       CONS(CLAUSE,clause_reductions(directive_module,directive_txt(d)),NIL));
  // private handling
  directive_clauses(d) = gen_nconc(directive_clauses(d),
				       CONS(CLAUSE,clause_private(directive_module,directive_txt(d)),NIL));
}

static statement do_outlining(directive d)
{
  statement call; 
  list body;
  loop loop_l;
  string new_module_name, id_suffix, lower, upper;
  entity new_module, index, low_e, up_e;
  expression low_range_expr, up_range_expr;
  loop_data data;

  pips_debug(1,"d = %p\n", d);

  pips_assert("do",type_directive_omp_parallel_do_p(directive_type(d))
	      ||  type_directive_omp_do_p(directive_type(d)));
  
  body = directive_body(d);

  pips_assert("1 statement",gen_length(body)==1);
  pips_assert("loop statement",statement_loop_p(STATEMENT(CAR(body))));

  new_module_name = directive_module_name(d);
  new_module = outlining_start(new_module_name);

  // loop bounds variables substitution
  loop_l = instruction_loop(statement_instruction(STATEMENT(CAR(body))));

  id_suffix = step_find_loop_range_suffix_id(strdup(concatenate(new_module_name, MODULE_SEP_STRING,module_local_name(loop_index(loop_l)),NULL)));
  lower=strdup(concatenate(entity_user_name(loop_index(loop_l)), RANGENAME_LOW_SUFFIX, id_suffix, NULL));
  upper=strdup(concatenate(entity_user_name(loop_index(loop_l)), RANGENAME_UP_SUFFIX, id_suffix, NULL));

  index = outlining_add_declaration(loop_index(loop_l));
  low_e = outlining_add_declaration(clone_scalar(lower, new_module, index));
  up_e = outlining_add_declaration(clone_scalar(upper, new_module, index));

  low_range_expr = range_lower(loop_range(loop_l));
  up_range_expr = range_upper(loop_range(loop_l));
  range_lower(loop_range(loop_l))=entity_to_expr(low_e);
  range_upper(loop_range(loop_l))=entity_to_expr(up_e);

  outlining_scan_block(gen_full_copy_list(body));
  
  outlining_add_argument(index,entity_to_expr(index));
  outlining_add_argument(low_e,low_range_expr);
  outlining_add_argument(up_e,up_range_expr);

  call = outlining_close();

  data = make_loop_data(index,low_e,up_e,expression_to_int(range_increment(loop_range(loop_l))));

  if (type_directive_omp_parallel_do_p(directive_type(d)))
    type_directive_omp_parallel_do(directive_type(d)) = CONS(LOOP_DATA,data,type_directive_omp_parallel_do(directive_type(d)));
  else
    type_directive_omp_do(directive_type(d)) = CONS(LOOP_DATA,data,type_directive_omp_do(directive_type(d)));
    
  if(statement_comments(call)!=empty_comments)
    {
      free(statement_comments(call));
      statement_comments(call) = empty_comments;
    }
  statement_label(call) = statement_label(STATEMENT(CAR(body)));

  pips_debug(1,"call = %p\n", call);
  return call;
}

directive make_directive_omp_parallel_do(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_PARALLEL_DO,""),make_type_directive_omp_parallel_do(NIL),NIL,NIL);

  pips_debug(1,"d = %p\n", d);

  return d;
}

bool is_begin_directive_omp_parallel_do(directive __attribute__ ((unused)) current,directive next)
{
  bool b;

  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_parallel_do_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_end_parallel_do(statement stmt)
{
  string new_name;
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name(SUFFIX_OMP_END_PARALLEL_DO,"");

  d = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_end_parallel_do(),NIL,NIL);

  pips_debug(1,"d = %p\n", d);

  return d;
}

bool is_end_directive_omp_end_parallel_do(directive current, directive next)
{
  bool b1, b2;

  pips_debug(1,"current = %p, next = %p\n", current, next);

  b1 = type_directive_omp_parallel_do_p(directive_type(current));
  b2 = type_directive_omp_end_parallel_do_p(directive_type(next));

  pips_debug(1,"b1 = %d, b2 = %d\n", b1, b2);
  return b1 && b2;
}

/* 
   retourner une sequence contenant le body du begin (en l'occurrence
   une liste) */

instruction handle_omp_parallel_do(directive begin,directive end)
{
  instruction instr;
  statement call;
  entity directive_module;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  call = do_outlining(begin);
  directive_module = call_function(statement_call(call));
  call = step_keep_directive_txt(begin, call, end);

  if (type_directive_none_p(directive_type(end)))
      instr = statement_instruction(make_block_statement(CONS(STATEMENT,call,gen_full_copy_list(directive_body(end)))));
  else
    {
      pips_assert("end_parallel_do directive",type_directive_omp_end_parallel_do_p(directive_type(end)));
      instr = statement_instruction(call);
      statement_instruction(call) = instruction_undefined;
      free_statement(call);
    }
  
  clause_handling(directive_module, begin);
  
  store_global_directives(directive_module,begin);

  // liberation temporaire
  //  free_directive(begin);
  free_directive(end);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

directive make_directive_omp_do(statement stmt)
{
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  d = make_directive(strdup(statement_to_directive_txt(stmt)),step_make_new_directive_module_name(SUFFIX_OMP_DO,""),make_type_directive_omp_do(NIL),NIL,NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_begin_directive_omp_do(directive __attribute__ ((unused)) current, directive next)
{
  bool b;
  pips_debug(1,"next = %p\n", next);

  b = type_directive_omp_do_p(directive_type(next));

  pips_debug(1,"b = %d\n", b);
  return b;
}

directive make_directive_omp_end_do(statement stmt)
{
  string new_name;
  directive d;

  pips_debug(1,"stmt = %p\n", stmt);

  new_name = step_make_new_directive_module_name(SUFFIX_OMP_END_DO, "");
  d = make_directive(strdup(statement_to_directive_txt(stmt)), new_name, make_type_directive_omp_end_do(), NIL, NIL);

  pips_debug(1,"d = %p\n", d);
  return d;
}

bool is_end_directive_omp_end_do(directive current, directive next)
{
  bool b1, b2;
    
  pips_debug(1,"current = %p, next = %p\n", current, next);

  b1 = type_directive_omp_end_do_p(directive_type(next));
  b2 = type_directive_omp_do_p(directive_type(current));

  pips_debug(1,"b1 = %d, b2 = %d\n", b1, b2);
  return b1 && b2;
}

instruction handle_omp_do(directive begin, directive end)
{
  instruction instr;
  statement call;
  entity directive_module;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);

  call = do_outlining(begin);
  directive_module = call_function(statement_call(call));

  call = step_keep_directive_txt(begin, call, end);

  if (type_directive_none_p(directive_type(end)))
    {
      /* Case where the end directive is not present.
	 In this case, the none directive contains the next statement.
	 Thus it is added to the instruction list.
      */
      
      instr = statement_instruction(make_block_statement(CONS(STATEMENT, call, gen_full_copy_list(directive_body(end)))));
    }
  else
    {
      pips_assert("end_do directive", type_directive_omp_end_do_p(directive_type(end)));
      instr = statement_instruction(call);
      statement_instruction(call) = instruction_undefined;
      free_statement(call);
    }

  clause_handling(directive_module, begin);

  store_global_directives(directive_module,begin);
  free_directive(end);

  pips_debug(1,"instr = %p\n", instr);
  return instr;
}

/*
  In a case of a loop statement only
  Update module name associated with the directive
  with the label of the loop
*/
bool update_label_do_directive_module_name(directive begin, directive end)
{
  string suffix;
  statement stmt;
  entity label;
  directive drt;
  string old_name;

  pips_debug(1,"begin = %p, end = %p\n", begin, end);
  pips_assert("directive none", type_directive_none_p(directive_type(end)));

  switch (type_directive_tag(directive_type(begin)))
    {
    case is_type_directive_omp_parallel_do:
      suffix = SUFFIX_OMP_PARALLEL_DO;
      break;
    case is_type_directive_omp_do:
      suffix = SUFFIX_OMP_DO;
      break;
    default:
      return FALSE;
    }

  stmt = STATEMENT(CAR(directive_body(end)));
  if (!statement_loop_p(stmt))
    return FALSE;

  /* Case of a loop statement only */
  label = loop_label(instruction_loop(statement_instruction(stmt)));
  if (entity_empty_label_p(label))
    return FALSE;
  
  drt = current_directives_pop();
  STEP_DEBUG_DIRECTIVE(2, "pop current_directives", drt);

  old_name = directive_module_name(drt);

  gen_clear_tabulated_element(gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING, old_name, NULL),entity_domain));

  free(old_name);
  directive_module_name(drt) = step_make_new_directive_module_name(suffix, label_local_name(label));
  current_directives_push(drt);

  pips_debug(1,"TRUE\n");
  return TRUE;
}


string directive_omp_do_to_string(directive d,bool close)
{
  pips_debug(1, "d=%p, close=%u\n",d,close);
  if (close)
    return strdup(END_DO_TEXT);
  else
    return strdup(DO_TEXT);
}

string directive_omp_parallel_do_to_string(directive d,bool close)
{
  pips_debug(1, "d=%p, close=%u\n",d,close);
  if (close)
    return strdup(END_PARALLEL_DO_TEXT);
  else
    return strdup(PARALLEL_DO_TEXT);
}
