/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

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

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "text-util.h"
#include "effects-simple.h"
#include "effects-generic.h"
#include "pipsdbm.h"
#include "misc.h"

#include "pointer_values.h"

/******************** MAPPINGS */

/* I don't know how to deal with these mappings if we have to analyse several modules
   at the same time when performing an interprocedural analysis.
   We may need a stack of mappings, or a global mapping for the whole program,
   and a temporary mapping to store the resources one module at a time
*/
GENERIC_GLOBAL_FUNCTION(pv, statement_cell_relations)
GENERIC_GLOBAL_FUNCTION(gen_pv, statement_cell_relations)
GENERIC_GLOBAL_FUNCTION(kill_pv, statement_effects)



/******************** PIPSDBM INTERFACES */

statement_cell_relations db_get_simple_pv(const char * module_name)
{
  return (statement_cell_relations) db_get_memory_resource(DBR_SIMPLE_POINTER_VALUES, module_name, true);
}

void db_put_simple_pv(const char * module_name, statement_cell_relations scr)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_POINTER_VALUES, module_name, (char*) scr);
}

list db_get_in_simple_pv(const char * module_name)
{
  return (list) cell_relations_list((cell_relations) db_get_memory_resource(DBR_IN_SIMPLE_POINTER_VALUES, module_name, true));
}

void db_put_in_simple_pv(const char * module_name, list l_pv)
{
  DB_PUT_MEMORY_RESOURCE(DBR_IN_SIMPLE_POINTER_VALUES, module_name, (char*) make_cell_relations(l_pv));
}

list db_get_out_simple_pv(const char * module_name)
{
  return cell_relations_list((cell_relations) db_get_memory_resource(DBR_OUT_SIMPLE_POINTER_VALUES, module_name, true));
}

void db_put_out_simple_pv(const char * module_name, list l_pv)
{
  DB_PUT_MEMORY_RESOURCE(DBR_OUT_SIMPLE_POINTER_VALUES, module_name, (char*) make_cell_relations(l_pv));
}

list db_get_initial_simple_pv(const char * module_name)
{
  return cell_relations_list((cell_relations) db_get_memory_resource(DBR_INITIAL_SIMPLE_POINTER_VALUES, module_name, true));
}

void db_put_initial_simple_pv(const char * module_name, list l_pv)
{
  DB_PUT_MEMORY_RESOURCE(DBR_INITIAL_SIMPLE_POINTER_VALUES, module_name, (char*) make_cell_relations(l_pv));
}

list db_get_program_simple_pv()
{
  return cell_relations_list((cell_relations) db_get_memory_resource(DBR_PROGRAM_SIMPLE_POINTER_VALUES, "", true));
}

void db_put_program_simple_pv(list l_pv)
{
  DB_PUT_MEMORY_RESOURCE(DBR_PROGRAM_SIMPLE_POINTER_VALUES, "", (char*) make_cell_relations(l_pv));
}




/******************** ANALYSIS CONTEXT */


pv_context make_simple_pv_context()
{
  pv_context ctxt;

  ctxt.initial_pointer_values_p = false;

  ctxt.db_get_pv_func = db_get_simple_pv;
  ctxt.db_put_pv_func = db_put_simple_pv;
  ctxt.db_get_in_pv_func = db_get_in_simple_pv;
  ctxt.db_put_in_pv_func = db_put_in_simple_pv;
  ctxt.db_get_out_pv_func = db_get_out_simple_pv;
  ctxt.db_put_out_pv_func = db_put_out_simple_pv;
  ctxt.db_get_initial_pv_func = db_get_initial_simple_pv;
  ctxt.db_put_initial_pv_func = db_put_initial_simple_pv;
  ctxt.db_get_program_pv_func = db_get_program_simple_pv;
  ctxt.db_put_program_pv_func = db_put_program_simple_pv;

  ctxt.make_pv_from_effects_func = make_simple_pv_from_simple_effects;
  ctxt.cell_preceding_p_func = simple_cell_preceding_p;
  ctxt.cell_reference_with_value_of_cell_reference_translation_func =
    simple_cell_reference_with_value_of_cell_reference_translation;
  ctxt.cell_reference_with_address_of_cell_reference_translation_func =
    simple_cell_reference_with_address_of_cell_reference_translation;
  ctxt.pv_composition_with_transformer_func = simple_pv_composition_with_transformer;
  ctxt.pvs_must_union_func = simple_pvs_must_union;
  ctxt.pvs_may_union_func = simple_pvs_may_union;
  ctxt.pvs_equal_p_func = simple_pvs_syntactically_equal_p;
  ctxt.stmt_stack = stack_make(statement_domain, 0, 0);
  return ctxt;
}


#define UNDEF abort

typedef void (*void_function)();
typedef gen_chunk* (*chunks_function)();
typedef list (*list_function)();
typedef bool (*bool_function)();
typedef descriptor (*descriptor_function)();
typedef statement_cell_relations (*statement_cell_relations_function)();
typedef statement_effects (*statement_effects_function)();
typedef cell_relation (*cell_relation_function)();

void reset_pv_context(pv_context *p_ctxt)
{
  p_ctxt->db_get_pv_func = (statement_cell_relations_function) UNDEF;
  p_ctxt->db_put_pv_func = (void_function) UNDEF;
/*   p_ctxt->db_get_gen_pv_func =(statement_cell_relations_function) UNDEF ; */
/*   p_ctxt->db_put_gen_pv_func = (void_function) UNDEF; */
/*   p_ctxt->db_get_kill_pv_func = (statement_effects_function) UNDEF; */
/*   p_ctxt->db_put_kill_pv_func = (void_function) UNDEF; */
  p_ctxt->make_pv_from_effects_func = (list_function) UNDEF;
  p_ctxt->pv_composition_with_transformer_func = (cell_relation_function) UNDEF;
  p_ctxt->pvs_must_union_func = (list_function) UNDEF;
  p_ctxt->pvs_may_union_func = (list_function) UNDEF;
  p_ctxt->pvs_equal_p_func = (bool_function) UNDEF;
}

void pv_context_statement_push(statement s, pv_context * ctxt)
{
  stack_push((void *) s, ctxt->stmt_stack);
}

void pv_context_statement_pop(pv_context * ctxt)
{
  (void) stack_pop( ctxt->stmt_stack);
}

statement pv_context_statement_head(pv_context * ctxt)
{
  return ((statement) stack_head(ctxt->stmt_stack));
}

/************* RESULTS HOOK */

pv_results make_pv_results()
{
  pv_results pv_res;
  pv_res.l_out = NIL;
  pv_res.result_paths = NIL;
  pv_res.result_paths_interpretations = NIL;
  return pv_res;
}

void free_pv_results_paths(pv_results *pv_res)
{
  gen_full_free_list(pv_res->result_paths);
  pv_res->result_paths = NIL;
  gen_full_free_list(pv_res->result_paths_interpretations);
  pv_res->result_paths_interpretations = NIL;
}

void print_pv_results(pv_results pv_res)
{
  fprintf(stderr, "l_out =");
  print_pointer_values(pv_res.l_out);
  list l_rp = pv_res.result_paths;
  list l_rpi = pv_res.result_paths_interpretations;

  if (!ENDP(l_rp))
    {
      fprintf(stderr, "result_paths are:\n");
      for(; !ENDP(l_rp); POP(l_rp), POP(l_rpi))
	{
	  effect eff = EFFECT(CAR(l_rp));
	  cell_interpretation ci = CELL_INTERPRETATION(CAR(l_rpi));
	  fprintf(stderr, "%s:",
		  cell_interpretation_value_of_p(ci)
		  ? "value of" : "address of");
	  (*effect_prettyprint_func)(eff);
	}
    }
  else
    fprintf(stderr, "result_path is undefined\n");
}

/******************** SOME UTILITIES (to move elsewhere?) */

list make_anywhere_anywhere_pvs()
{
  cell anywhere_c = make_cell_reference(make_reference(entity_all_locations(), NIL));
  cell_relation pv = make_address_of_pointer_value(anywhere_c,
						   copy_cell(anywhere_c),
						   is_approximation_may,
						   make_descriptor_none());
  return (CONS(CELL_RELATION, pv, NIL));
}

/******************** LOCAL FUNCTIONS DECLARATIONS */

static
list sequence_to_post_pv(sequence seq, list l_in, pv_context *ctxt);

static
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt);

static
list declarations_to_post_pv(list l_decl, list l_in, pv_context *ctxt);

static
list declaration_to_post_pv(entity e, list l_in, pv_context *ctxt);

static
list instruction_to_post_pv(instruction inst, list l_in, pv_context *ctxt);

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt);

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt);

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt);

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt);

static
list unstructured_to_post_pv(unstructured u, list l_in, pv_context *ctxt);


static
void call_to_post_pv(call c, list l_in, pv_results *pv_res, pv_context *ctxt);




/**************** MODULE ANALYSIS *************/

static
list sequence_to_post_pv(sequence seq, list l_in, pv_context *ctxt)
{
  list l_cur = l_in;
  list l_locals = NIL;
  pips_debug(1, "begin\n");
  FOREACH(STATEMENT, stmt, sequence_statements(seq))
    {
      ifdebug(2){

	pips_debug(2, "dealing with statement");
	print_statement(stmt);
	pips_debug_pvs(2, "l_cur =", l_cur);
      }
      /* keep local variables in declaration reverse order */
      if (declaration_statement_p(stmt))
	{
	  pv_context_statement_push(stmt, ctxt);
	  if(bound_pv_p(stmt))
	    update_pv(stmt, make_cell_relations(gen_full_copy_list(l_cur)));
	  else
	    store_pv(stmt, make_cell_relations(gen_full_copy_list(l_cur)));
	  FOREACH(ENTITY, e, statement_declarations(stmt))
	    {
	      type e_type = entity_basic_concrete_type(e);
	      storage e_storage = entity_storage(e);
	      /* beware don't push static variables and non pointer variables. */
	      if (storage_ram_p(e_storage)
		  && ! static_area_p(ram_section(storage_ram(e_storage)))
		  && ! type_fundamental_basic_p(e_type)
		  && basic_concrete_type_leads_to_pointer_p(e_type))
		l_locals = CONS(ENTITY, e, l_locals);
	      l_cur = declaration_to_post_pv(e, l_cur, ctxt);
	    }
	  //store_gen_pv(stmt, make_cell_relations(NIL));
	  //store_kill_pv(stmt, make_effects(NIL));
	  pv_context_statement_pop(ctxt);
	}
      else
	l_cur = statement_to_post_pv(stmt, l_cur, ctxt);

    }

  /* don't forget to eliminate local declarations on exit */
  /* ... */
  if (!ENDP(l_locals))
    {
      pips_debug(5, "eliminating local variables\n");
      expression rhs_exp =
	entity_to_expression(undefined_pointer_value_entity());

      FOREACH(ENTITY, e, l_locals)
	{
	  pips_debug(5, "dealing with %s\n", entity_name(e));
	  pv_results pv_res = make_pv_results();
	  pointer_values_remove_var(e, false, l_cur, &pv_res, ctxt);
	  l_cur= pv_res.l_out;
	  free_pv_results_paths(&pv_res);
	}
      free_expression(rhs_exp);
    }

  pips_debug_pvs(2, "returning:", l_cur);
  pips_debug(1, "end\n");
  return (l_cur);
}

static
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pv_context_statement_push(stmt, ctxt);
  pips_debug_pvs(2, "input pvs:", l_in);

  if(bound_pv_p(stmt))
    update_pv(stmt, make_cell_relations(gen_full_copy_list(l_in)));
  else
    store_pv(stmt, make_cell_relations(gen_full_copy_list(l_in)));

  if (declaration_statement_p(stmt))
    {
      list l_decl = statement_declarations(stmt);
      l_out = declarations_to_post_pv(l_decl, l_in, ctxt);
    }
  else
    {
      l_out = instruction_to_post_pv(statement_instruction(stmt), l_in, ctxt);
    }

  pips_debug_pvs(2, "before composition_with_transformer:", l_out);
  l_out = pvs_composition_with_transformer(l_out, transformer_undefined, ctxt);

  //store_gen_pv(stmt, make_cell_relations(NIL));
  //store_kill_pv(stmt, make_effects(NIL));
  pips_debug_pvs(2, "returning:", l_out);
  pips_debug(1, "end\n");
  pv_context_statement_pop(ctxt);

  return (l_out);
}

static
list declarations_to_post_pv(list l_decl, list l_in, pv_context *ctxt)
{
  list l_out = l_in;
  pips_debug(1, "begin\n");

  FOREACH(ENTITY, e, l_decl)
    {
      /* well, not exactly, we must take kills into account */
      l_out = gen_nconc(l_out, declaration_to_post_pv(e, l_out, ctxt));
    }
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}


static
list declaration_to_post_pv(entity e, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  type e_type = entity_basic_concrete_type(e);

  if (type_variable_p(e_type) && !typedef_entity_p(e)
      && !type_fundamental_basic_p(e_type)
      && basic_concrete_type_leads_to_pointer_p(e_type))
    {
      pips_debug(1, "begin\n");
      pips_debug(1, "entity %s basic concrete type %s %s\n",
	     entity_name(e), type_to_string(e_type), string_of_type(e_type));
      expression lhs_exp = entity_to_expression(e);
      expression rhs_exp = expression_undefined;
      bool free_rhs_exp = false; /* turned to true if rhs_exp is built and
				    must be freed */

      value v_init = entity_initial(e);
      bool static_p = (!ctxt->initial_pointer_values_p) && entity_static_variable_p(e);

      /* in case of a local static variable, and when not dealing with the initial analysis,
	 we should generate a stub sink */

      if (v_init == value_undefined)
	{
	  pips_debug(2, "undefined inital value\n");

	  if (std_file_entity_p(e))
	    {
	      pips_debug(2, "standard file\n");
	      expression tmp_exp = entity_to_expression(std_file_entity_to_pointed_file_entity(e));
	      rhs_exp = make_address_of_expression(tmp_exp);
	      free_expression(tmp_exp);
	    }
	  else if (static_p)
	    /* when no initialization is provided for pointer static variables,
	       or aggregate variables which recursively have pointer fields,
	       then all pointers are initialized to NULL previous to program
	       execution.
	    */
	    rhs_exp = entity_to_expression(null_pointer_value_entity());
	  else
	    rhs_exp = entity_to_expression(undefined_pointer_value_entity());
	  free_rhs_exp = true;
	}
      else if (std_file_entity_p(e))
	{
	  pips_debug(2, "standard file\n");
	  expression tmp_exp = entity_to_expression(std_file_entity_to_pointed_file_entity(e));
	  rhs_exp = make_address_of_expression(tmp_exp);
	  free_expression(tmp_exp);
	  free_rhs_exp = true;
	}
      else
	{
	  switch (value_tag(v_init))
	    {
	    case is_value_expression:
	      rhs_exp = value_expression(v_init);
	      break;
	    case is_value_unknown:
	      pips_debug(2, "unknown inital value\n");
	      rhs_exp = expression_undefined;
	      break;
	    case is_value_code:
	    case is_value_symbolic:
	    case is_value_constant:
	    case is_value_intrinsic:
	    default:
	      pips_internal_error("unexpected tag");
	    }
	}

      pv_results pv_res = make_pv_results();
      assignment_to_post_pv(lhs_exp, static_p, rhs_exp, true, l_in, &pv_res, ctxt);
      l_out = pv_res.l_out;
      free_pv_results_paths(&pv_res);

      free_expression(lhs_exp);
      if (free_rhs_exp) free_expression(rhs_exp);
      pips_debug_pvs(2, "returning:", l_out);
      pips_debug(1, "end\n");
    }
  else
    l_out = l_in;
  return (l_out);
}

static
list instruction_to_post_pv(instruction inst, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");

  switch(instruction_tag(inst))
    {
    case is_instruction_sequence:
      l_out = sequence_to_post_pv(instruction_sequence(inst), l_in, ctxt);
      break;
    case is_instruction_test:
      l_out = test_to_post_pv(instruction_test(inst), l_in, ctxt);
      break;
    case is_instruction_loop:
      l_out = loop_to_post_pv(instruction_loop(inst), l_in, ctxt);
      break;
    case is_instruction_whileloop:
      l_out = whileloop_to_post_pv(instruction_whileloop(inst), l_in, ctxt);
      break;
    case is_instruction_forloop:
      l_out = forloop_to_post_pv(instruction_forloop(inst), l_in, ctxt);
      break;
    case is_instruction_unstructured:
      l_out = unstructured_to_post_pv(instruction_unstructured(inst), l_in, ctxt);
      break;
    case is_instruction_expression:
      {
	pv_results pv_res = make_pv_results();
	expression_to_post_pv(instruction_expression(inst), l_in, &pv_res, ctxt);
	l_out = pv_res.l_out;
	free_pv_results_paths(&pv_res);
      }
      break;
    case is_instruction_call:
      {
	pv_results pv_res = make_pv_results();
	call_to_post_pv(instruction_call(inst), l_in, &pv_res, ctxt);
	l_out = pv_res.l_out;
	free_pv_results_paths(&pv_res);
      }
      break;
    case is_instruction_goto:
      pips_internal_error("unexpected goto in pointer values analyses");
      break;
    case is_instruction_multitest:
      pips_internal_error("unexpected multitest in pointer values analyses");
      break;
    default:
      pips_internal_error("unknown instruction tag");
    }
  pips_debug_pvs(2, "returning:", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");

  expression t_cond = test_condition(t);
  statement t_true = test_true(t);
  statement t_false = test_false(t);

  pv_results pv_res = make_pv_results();
  expression_to_post_pv(t_cond, l_in, &pv_res, ctxt);

  list l_in_branches = pv_res.l_out;

  list l_out_true = statement_to_post_pv(t_true, gen_full_copy_list(l_in_branches), ctxt);
  list l_out_false = statement_to_post_pv(t_false, l_in_branches, ctxt);

  l_out = (*ctxt->pvs_may_union_func)(l_out_true, l_out_false);

  free_pv_results_paths(&pv_res);

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

#define PV_NB_MAX_ITER_FIX_POINT 3

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  range r = loop_range(l);
  list l_in_cur = l_in;
  statement body = loop_body(l);
  pips_debug(1, "begin\n");

  /* first loop range is evaluated */
  pv_results pv_res = make_pv_results();
  range_to_post_pv(r, l_in_cur, &pv_res, ctxt);
  free_pv_results_paths(&pv_res);
  l_in_cur = pv_res.l_out;
  list l_saved = gen_full_copy_list(l_in_cur);

  /* then, the loop body is executed if and only if the upper bound
     is greater than the lower bound, else the loop body is only possibly
     executed.
  */

  /* as a first approximation, we perform no test on the loop bounds,
     and thus assume that the loop body is only possibly executed
  */
  int i = 0;
  bool fix_point_reached = false;
  l_out = pv_res.l_out;
  do
    {
      pips_debug(3, "fix point iteration number %d.\n", i+1);
      list l_iter_in = gen_full_copy_list(l_out);
      l_out = statement_to_post_pv(body, l_out, ctxt);

      /* this iteration may never be executed :*/
      l_out = (*ctxt->pvs_may_union_func)(l_out, gen_full_copy_list(l_iter_in));
      i++;
      fix_point_reached = (l_out == l_iter_in)
	|| (*ctxt->pvs_equal_p_func)(l_iter_in, l_out);
      pips_debug(3, "fix point %s reached\n", fix_point_reached? "":"not");
      gen_full_free_list(l_iter_in);
    }
  while(i<PV_NB_MAX_ITER_FIX_POINT && !fix_point_reached);

  if (!fix_point_reached)
    {
      pv_results pv_res_failed = make_pv_results();
      effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
      list l_anywhere = CONS(EFFECT, anywhere_eff, NIL);
      list l_kind = CONS(CELL_INTERPRETATION,
			 make_cell_interpretation_address_of(), NIL);
      single_pointer_assignment_to_post_pv(anywhere_eff, l_anywhere, l_kind,
					   false, l_saved,
					   &pv_res_failed, ctxt);
      l_out =  pv_res_failed.l_out;
      free_pv_results_paths(&pv_res_failed);
      free_effect(anywhere_eff);
      gen_free_list(l_anywhere);
      gen_full_free_list(l_kind);

      /* now update input pointer values of inner loop statements */
      l_in_cur = gen_full_copy_list(l_out);
      list l_tmp = statement_to_post_pv(body, l_in_cur, ctxt);
      gen_full_free_list(l_tmp);
    }
  else
    gen_full_free_list(l_saved);

  pips_debug_pvs(1, "end with l_out =\n", l_out);
  return (l_out);
}

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  list l_in_cur = NIL;
  list l_saved = gen_full_copy_list(l_in); /* in case fix point is not reached */
  expression cond = whileloop_condition(l);
  bool before_p = evaluation_before_p(whileloop_evaluation(l));
  statement body = whileloop_body(l);
  pips_debug(1, "begin\n");

  int i = 1;
  bool fix_point_reached = false;
  l_out = l_in;
  do
    {
      pips_debug(3, "fix point iteration number %d.\n", i);
      list l_iter_in = gen_full_copy_list(l_out);
      l_in_cur = l_out;

      if(before_p)
	{
	  pv_results pv_res_cond = make_pv_results();
	  expression_to_post_pv(cond, l_in_cur, &pv_res_cond, ctxt);
	  l_in_cur = pv_res_cond.l_out;
	  free_pv_results_paths(&pv_res_cond);
	}

      l_out = statement_to_post_pv(body, l_in_cur, ctxt);

      if(!before_p)
	{
	  pv_results pv_res_cond = make_pv_results();
	  expression_to_post_pv(cond, l_out, &pv_res_cond, ctxt);
	  l_out = pv_res_cond.l_out;
	  free_pv_results_paths(&pv_res_cond);
	}

      /* this iteration may never be executed :*/
      if (i!=1 || before_p)
	  l_out = (*ctxt->pvs_may_union_func)(l_out,
					      gen_full_copy_list(l_iter_in));
      else
	{
	  l_in = gen_full_copy_list(l_out);
	}

      i++;
      fix_point_reached = (l_out == l_iter_in)
	|| (*ctxt->pvs_equal_p_func)(l_iter_in, l_out);
      pips_debug(3, "fix point %s reached\n", fix_point_reached? "":"not");
      gen_full_free_list(l_iter_in);
    }
  while(i<=PV_NB_MAX_ITER_FIX_POINT && !fix_point_reached);

  if (!fix_point_reached)
    {
      pv_results pv_res_failed = make_pv_results();
      effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
      list l_anywhere = CONS(EFFECT, anywhere_eff, NIL);
      list l_kind = CONS(CELL_INTERPRETATION,
			 make_cell_interpretation_address_of(), NIL);
      single_pointer_assignment_to_post_pv(anywhere_eff, l_anywhere, l_kind,
					   false, l_saved,
					   &pv_res_failed, ctxt);
      l_out =  pv_res_failed.l_out;
      free_pv_results_paths(&pv_res_failed);
      free_effect(anywhere_eff);
      gen_free_list(l_anywhere);
      gen_full_free_list(l_kind);

      /* now update input pointer values of inner loop statements */
      l_in_cur = gen_full_copy_list(l_out);
      if(before_p)
	{
	  pv_results pv_res_cond = make_pv_results();
	  expression_to_post_pv(cond, l_in_cur, &pv_res_cond, ctxt);
	  l_in_cur = pv_res_cond.l_out;
	  free_pv_results_paths(&pv_res_cond);
	}

      list l_tmp = statement_to_post_pv(body, l_in_cur, ctxt);

      if(!before_p)
	{
	  pv_results pv_res_cond = make_pv_results();
	  expression_to_post_pv(cond, l_tmp, &pv_res_cond, ctxt);
	  free_pv_results_paths(&pv_res_cond);
	  gen_full_free_list(pv_res_cond.l_out);
	}
      else
	gen_full_free_list(l_tmp);
    }
  else
    gen_full_free_list(l_saved);

  pips_debug_pvs(2, "returning:", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  list l_in_cur = NIL;
  pips_debug(1, "begin\n");

  expression init = forloop_initialization(l);
  expression cond = forloop_condition(l);
  expression incr = forloop_increment(l);
  statement body = forloop_body(l);

  /* First, the initialization is always evaluatated */
  pv_results pv_res_init = make_pv_results();
  expression_to_post_pv(init, l_in, &pv_res_init, ctxt);
  l_in_cur = pv_res_init.l_out;
  l_in = gen_full_copy_list(l_in_cur); /* saved in case fix point is not reached */
  free_pv_results_paths(&pv_res_init);

  int i = 1;
  bool fix_point_reached = false;
  l_out = l_in_cur;
  do
    {
      pips_debug(3, "fix point iteration number %d.\n", i);
      list l_iter_in = gen_full_copy_list(l_out);
      l_in_cur = l_out;

      /* condition is evaluated before each iteration */
      pv_results pv_res_cond = make_pv_results();
      expression_to_post_pv(cond, l_in_cur, &pv_res_cond, ctxt);
      l_in_cur = pv_res_cond.l_out;
      free_pv_results_paths(&pv_res_cond);

      l_in_cur = statement_to_post_pv(body, l_in_cur, ctxt);

      /* increment expression is evaluated at the end of each iteration */
      pv_results pv_res_incr = make_pv_results();
      expression_to_post_pv(incr, l_in_cur, &pv_res_incr, ctxt);
      l_in_cur = pv_res_incr.l_out;
      free_pv_results_paths(&pv_res_incr);

      /* this iteration may never be executed :*/
      l_out = (*ctxt->pvs_may_union_func)(l_in_cur,
					  gen_full_copy_list(l_iter_in));
      i++;
      fix_point_reached = (l_out == l_iter_in)
	|| (*ctxt->pvs_equal_p_func)(l_iter_in, l_out);
      pips_debug(3, "fix point %s reached\n", fix_point_reached? "":"not");
      gen_full_free_list(l_iter_in);
    }
  while(i<=PV_NB_MAX_ITER_FIX_POINT && !fix_point_reached);

  if (!fix_point_reached)
    {
      pv_results pv_res_failed = make_pv_results();
      effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
      list l_anywhere = CONS(EFFECT, anywhere_eff, NIL);
      list l_kind = CONS(CELL_INTERPRETATION,
			 make_cell_interpretation_address_of(), NIL);
      single_pointer_assignment_to_post_pv(anywhere_eff, l_anywhere, l_kind,
					   false, l_in,
					   &pv_res_failed, ctxt);
      l_out =  pv_res_failed.l_out;
      free_pv_results_paths(&pv_res_failed);
      free_effect(anywhere_eff);
      gen_free_list(l_anywhere);
      gen_full_free_list(l_kind);

      /* now update input pointer values of inner loop statements */
      l_in_cur = gen_full_copy_list(l_out);
      pv_results pv_res_cond = make_pv_results();
      expression_to_post_pv(cond, l_in_cur, &pv_res_cond, ctxt);
      l_in_cur = pv_res_cond.l_out;
      free_pv_results_paths(&pv_res_cond);

      list l_tmp = statement_to_post_pv(body, l_in_cur, ctxt);

      gen_full_free_list(l_tmp);
    }


  pips_debug_pvs(2, "returning:", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}


static
list unstructured_to_post_pv(unstructured unstr, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  control entry_ctrl = unstructured_entry( unstr );
  statement entry_node = control_statement(entry_ctrl);

  if(control_predecessors(entry_ctrl) == NIL && control_successors(entry_ctrl) == NIL)
    {
      /* there is only one statement in u; */
      pips_debug(6, "unique node\n");
      l_out = statement_to_post_pv(entry_node, l_in, ctxt);
    }
  else
    {
      pips_user_warning("Pointer analysis for unstructured part of code not yet fully implemented:\n"
			"Consider restructuring your code\n");
      list l_in_anywhere = make_anywhere_anywhere_pvs();
      list blocs = NIL ;
      CONTROL_MAP(c, {
	  list l_out = statement_to_post_pv(control_statement(c), gen_full_copy_list(l_in_anywhere), ctxt);
	  gen_full_free_list(l_out);
	},
	unstructured_exit( unstr ), blocs) ;
      gen_free_list(blocs);
      l_out = make_anywhere_anywhere_pvs();
    }
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

void range_to_post_pv(range r, list l_in, pv_results * pv_res, pv_context *ctxt)
{
    expression el = range_lower(r);
    expression eu = range_upper(r);
    expression ei = range_increment(r);

    pips_debug(1, "begin\n");

    expression_to_post_pv(el, l_in, pv_res, ctxt);
    expression_to_post_pv(eu, pv_res->l_out, pv_res, ctxt);
    expression_to_post_pv(ei, pv_res->l_out, pv_res, ctxt);

    free_pv_results_paths(pv_res);

    pips_debug_pvs(1, "end with pv_res->l_out:\n", pv_res->l_out);
}

void expression_to_post_pv(expression exp, list l_in, pv_results * pv_res, pv_context *ctxt)
{
  if (expression_undefined_p(exp))
    {
      pips_debug(1, "begin for undefined expression, returning undefined pointer_value\n");
      pv_res->l_out = l_in;
      pv_res->result_paths = CONS(EFFECT, make_effect(make_undefined_pointer_value_cell(),
						      make_action_write_memory(),
						      make_approximation_exact(),
						      make_descriptor_none()),NIL);
      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
						  make_cell_interpretation_value_of(), NIL);
    }
  else
    {
      pips_debug(1, "begin for expression : %s\n",
		 words_to_string(words_expression(exp,NIL)));

      syntax exp_syntax = expression_syntax(exp);

      switch(syntax_tag(exp_syntax))
	{
	case is_syntax_reference:
	  pips_debug(5, "reference case\n");
	  reference exp_ref = syntax_reference(exp_syntax);
	  if (same_string_p(entity_local_name(reference_variable(exp_ref)), "NULL"))
	    {
	      pv_res->result_paths = CONS(EFFECT, make_effect(make_null_pointer_value_cell(),
							     make_action_read_memory(),
							     make_approximation_exact(),
							     make_descriptor_none()), NIL);
	      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
							  make_cell_interpretation_value_of(),
							  NIL);
	    }
	  else
	    {
	      pv_res->result_paths = CONS(EFFECT,
					 (*reference_to_effect_func)
					 (copy_reference(exp_ref),
					  make_action_write_memory(),
					  false), NIL);
	      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
							  make_cell_interpretation_value_of(),
							  NIL);
	    }
	  /* we assume no effects on aliases due to subscripts evaluations for the moment */
	  pv_res->l_out = l_in;
	  break;
	case is_syntax_range:
	  pips_internal_error("not yet implemented");
	  break;
	case is_syntax_call:
	  {
	    call_to_post_pv(syntax_call(exp_syntax), l_in, pv_res, ctxt);
	    break;
	  }
	case is_syntax_cast:
	  {
	    expression_to_post_pv(cast_expression(syntax_cast(exp_syntax)),
				  l_in, pv_res, ctxt);
	    pips_debug(5, "cast case\n");
	  break;
	  }
	case is_syntax_sizeofexpression:
	  {
	    /* we assume no effects on aliases due to sizeof argument expression
	       for the moment */
	    pv_res->l_out = l_in;
	    break;
	  }
	case is_syntax_subscript:
	  {
	    pips_debug(5, "subscript case\n");
	    /* aborts if there are calls in subscript expressions */
	    list l_eff = NIL;
	    list l_tmp = generic_proper_effects_of_complex_address_expression(exp, &l_eff, true);
	    gen_full_free_list(l_tmp);
	    FOREACH(EFFECT, eff, l_eff)
	      {
		pv_res->result_paths = CONS(EFFECT, eff, NIL);
		pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
							    make_cell_interpretation_value_of(),
							    NIL);
	      }
	    gen_free_list(l_eff); /* free the spine */
	    /* we assume no effects on aliases due to subscripts evaluations for the moment */
	    pv_res->l_out = l_in;
	    break;
	  }
	case is_syntax_application:
	  pips_internal_error("not yet implemented");
	  break;
	case is_syntax_va_arg:
	  {
	    pips_internal_error("not yet implemented");
	    break;
	  }
	default:
	  pips_internal_error("unexpected tag %d", syntax_tag(exp_syntax));
	}
    }

  pips_debug_pv_results(1, "end with pv_results =\n", *pv_res);
  pips_debug(1, "end\n");
  return;
}



static void
external_call_to_post_pv(call c, list l_in, pv_results *pv_res, pv_context *ctxt)
{
  entity func = call_function(c);
  list func_args = call_arguments(c);
  pips_debug(1, "begin for %s\n", entity_local_name(func));

  /* the expression that denotes the called function and the arguments
     are evaluated; then there is a sequence point, and the function is called
  */

  /* Arguments evaluation: we don't use expressions_to_post_pv because
     we want to accumulate the results paths
  */

  list l_cur = l_in;
  pv_results pv_res_exp_eval = make_pv_results();
  FOREACH(EXPRESSION, arg, func_args)
    {
      pv_results pv_res_cur = make_pv_results();
      expression_to_post_pv(arg, l_cur, &pv_res_cur, ctxt);
      l_cur = pv_res_cur.l_out;
      pv_res_exp_eval.result_paths = gen_nconc(pv_res_exp_eval.result_paths, pv_res_cur.result_paths);
      pv_res_exp_eval.result_paths_interpretations = gen_nconc(pv_res_exp_eval.result_paths_interpretations,
						      pv_res_cur.result_paths_interpretations);
    }

  /* Function call: we generate abstract_location targets for all
     possible effects from arguments. We should also do the same for
     all global variables in the current compilation unit...
  */
  list l_ci = pv_res_exp_eval.result_paths_interpretations;
  FOREACH(EFFECT, real_arg_eff, pv_res_exp_eval.result_paths)
    {
      pips_debug_effect(3, "considering effect:\n", real_arg_eff);
      cell_interpretation ci = CELL_INTERPRETATION(CAR(l_ci));
      bool add_eff = cell_interpretation_address_of_p(ci);
      bool to_be_freed = false;
      type t = cell_to_type(effect_cell(real_arg_eff), &to_be_freed);

      // generate all possible effects on pointers from the actual parameter
      list lw = generic_effect_generate_all_accessible_paths_effects_with_level(real_arg_eff,
										t,
										'w',
										add_eff,
										10, /* to avoid too long paths until GAPS are handled */
										true); /* only pointers */
      if (to_be_freed) free_type(t);
      pips_debug_effects(3, "effects to be killed:\n", lw);

      if (!ENDP(lw))
	{
	  // potentially kill these paths and generate abstract location targets (currently anywhere)
	  effects_to_may_effects(lw);
	  list l_abstract_eff = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);
	  list l_ci_abstract_eff = CONS(CELL_INTERPRETATION, make_cell_interpretation_address_of(), NIL);
	  FOREACH(EFFECT, eff, lw)
	    {
	      single_pointer_assignment_to_post_pv(eff, l_abstract_eff, l_ci_abstract_eff,
						   false, l_cur,
						   pv_res, ctxt);
	      l_cur = pv_res->l_out;
	    }

	  gen_full_free_list(lw);
	}
      POP(l_ci);
    }
  pv_res->l_out = l_cur;

  /* Return value: test the type of the result path : if it's a
     pointer type, generate an abstract location target of the pointed
     type, else it's an empty set/list.
  */
  type func_res_t = compute_basic_concrete_type(functional_result(type_functional(entity_type(func))));
  if (pointer_type_p(func_res_t))
    {
      pv_res->result_paths = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()),
				  NIL);
      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
						  make_cell_interpretation_address_of(),
						  NIL);
    }
  else
    {
      pv_res->result_paths = NIL;
      pv_res->result_paths_interpretations = NIL;
    }
  free_type(func_res_t);

  pips_debug_pvs(2, "returning pv_res->l_out:", pv_res->l_out);
  pips_debug(1, "end\n");
  return;
}

static
void call_to_post_pv(call c, list l_in, pv_results *pv_res, pv_context *ctxt)
{
  entity func = call_function(c);
  value func_init = entity_initial(func);
  tag t = value_tag(func_init);
  list func_args = call_arguments(c);
  type func_type = ultimate_type(entity_type(func));

  pips_debug(1, "begin for %s\n", entity_local_name(func));

  if(type_functional_p(func_type))
    {
      switch (t)
	{
	case is_value_code:
	  external_call_to_post_pv(c, l_in, pv_res, ctxt);
	  break;

	case is_value_intrinsic:
	  pips_debug(5, "intrinsic function\n");
	  intrinsic_to_post_pv(func, func_args, l_in, pv_res, ctxt);
	  break;

	case is_value_symbolic:
	  pips_debug(5, "symbolic\n");
	  pv_res->l_out = l_in;
	  break;

	case is_value_constant:
	  pips_debug(5, "constant\n");
	  pv_res->l_out = l_in;

	  constant func_const = value_constant(func_init);
	  /* We should be here only in case of a pointer value rhs, and the value should be 0 */
	  if (constant_int_p(func_const) && (constant_int(func_const) == 0))
	    {
	      /* use approximation_exact to be consistent with effects,
		 should be approximation_exact */
	      pv_res->result_paths = CONS(EFFECT,
					  make_effect(make_null_pointer_value_cell(),
						      make_action_read_memory(),
						      make_approximation_exact(),
						      make_descriptor_none()),
					  NIL);
	      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
							  make_cell_interpretation_value_of(),
							  NIL);
	    }
	  else
	    {
	      type tt = functional_result(type_functional(func_type));
	      if (type_variable_p(tt))
		{
		  variable v = type_variable(tt);
		  basic b = variable_basic(v);
		  if (basic_string_p(b))/* constant strings */
		    {
		      /* not generic here */
		      effect eff = make_effect(make_cell_reference(make_reference(func, NIL)),
					       make_action_read_memory(),
					       make_approximation_exact(),
					       make_descriptor_none());
		      effect_add_dereferencing_dimension(eff);
		      pv_res->result_paths = CONS(EFFECT, eff, NIL);
		      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
							  make_cell_interpretation_address_of(),
							  NIL);
		    }
		}
	    }
	  pv_res->l_out = l_in;
	  break;

	case is_value_unknown:
	  pips_internal_error("unknown function ");
	  break;

	default:
	  pips_internal_error("unknown tag %d", t);
	  break;
	}
    }
  else if(type_variable_p(func_type))
    {
      pips_internal_error("not yet implemented");
    }
  else if(type_statement_p(func_type))
    {
      pips_internal_error("not yet implemented");
    }
  else
    {
      pips_internal_error("Unexpected case");
    }

  pips_debug_pvs(2, "returning pv_res->l_out:", pv_res->l_out);
  pips_debug(1, "end\n");
  return;
}



/*
  @brief returns in pv_res the effects of a single pointer assignment on pointer values
  @param lhs_eff is the left hand side path of the assignment
  @param l_rhs_eff is a list of rhs paths corresponding to the rhs
  @param l_rhs_kind is the list of rhs paths interpretations corresponding to elements of l_rhs_eff
  @param l_in is a list of the input pointer values
  @param pv_res is the struture holding the output result
  @param ctxt gives the functions specific to the kind of pointer values to be
          computed.
 */
void single_pointer_assignment_to_post_pv(effect lhs_eff,
					  list l_rhs_eff, list l_rhs_kind,
					  bool declaration_p, list l_in,
					  pv_results *pv_res, pv_context *ctxt)
{
  list l_out = NIL;
  list l_aliased = NIL;
  list l_kill = NIL;
  list l_gen = NIL;

  pips_debug_effect(1, "begin for lhs_eff =", lhs_eff);
  pips_debug(1, "and l_rhs_eff:\n");
  ifdebug(1)
    {
      list l_tmp = l_rhs_kind;
      FOREACH(EFFECT, rhs_eff, l_rhs_eff)
	{
	  cell_interpretation ci = CELL_INTERPRETATION(CAR(l_tmp));
	  pips_debug(1, "%s of:\n", cell_interpretation_value_of_p(ci)?"value": "address");
	  pips_debug_effect(1, "\t", rhs_eff);
	  POP(l_tmp);
	}
    }
  pips_debug_pvs(1, "and l_in =", l_in);
  bool anywhere_lhs_p = false;

  /* First search for all killed paths */
  /* we could be more precise/generic on abstract locations */
  if (anywhere_effect_p(lhs_eff))
    {
      pips_assert("we cannot have an anywhere lhs for a declaration\n", !declaration_p);
      pips_debug(3, "anywhere lhs\n");
      anywhere_lhs_p = true;
      l_kill = CONS(EFFECT, copy_effect(lhs_eff), NIL);
      l_aliased = l_kill;
    }
  else
    {
      if (!declaration_p) /* no aliases for a newly declared entity */
	{
	  l_aliased = effect_find_aliased_paths_with_pointer_values(lhs_eff, l_in, ctxt);
	  if (!ENDP(l_aliased) && anywhere_effect_p(EFFECT(CAR(l_aliased))))
	    {
	      pips_debug(3, "anywhere lhs (from aliases)\n");
	      anywhere_lhs_p = true;
	      l_kill = l_aliased;
	    }
	  else if (!ENDP(l_aliased) && ((int) gen_length(l_aliased) == 1)
		   && (null_pointer_value_effect_p(EFFECT(CAR(l_aliased)))))
	    {
	      // dereferencing a null pointer is considered as undefined by the standard
	      // with gcc (without any option) the compiler does not complain, but the execution aborts
	      // I make the choice here that if the pointer value is exactly NULL, then
	      // the program abort; hence there is no pointer anymore and the pointer values list is empty.
	      pips_user_warning("null pointer is dereferenced on lhs(%s)\n",
				effect_to_string(lhs_eff));
	      gen_full_free_list(l_in);
	      l_in = NIL;
	      l_kill = NIL;
	    }
	  else if (!ENDP(l_aliased) && ((int) gen_length(l_aliased) == 1)
		   && (undefined_pointer_value_effect_p(EFFECT(CAR(l_aliased)))))
		{
		  // dereferencing a non-initialized pointer is considered as undefined by the standard
		  // However, with gcc (without any option), the compiler does not complain,
		  // and the execution is sometimes correct.
		  // I make the choice here that if the pointer value is (exactly) undefined
		  // then the program does not necessarily abort;
		  // as I can keep dereferencements in pointer values (I'm not limited
		  // to constant paths), I still generate a kill and a gen. BC.
		  pips_user_warning("undefined pointer is dereferenced on lhs(%s)\n",
				    effect_to_string(lhs_eff));
		  l_kill = CONS(EFFECT, copy_effect(lhs_eff), NIL);
		}
	  else
	    {
	      /* if lhs_eff is a may-be-killed, then all aliased effects are also
		 may-be-killed effects */
	      if (effect_may_p(lhs_eff))
		{
		  pips_debug(3, "may lhs effect, changing all aliased effects to may\n");
		  effects_to_may_effects(l_aliased);
		}
	      l_kill = CONS(EFFECT, copy_effect(lhs_eff), l_aliased);
	    }
	}
      else
	{
	  l_kill = CONS(EFFECT, copy_effect(lhs_eff), NIL);
	}
    }

  pips_debug_effects(2, "l_kill =", l_kill);

  if (anywhere_lhs_p)
    {
      free_pv_results_paths(pv_res);
      pv_res->result_paths = l_aliased;
      pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION, make_cell_interpretation_address_of(), NIL);

      /* we must find in l_in all pointers p and generate p == rhs for all rhs if p is
	 of a type compatible with rhs, and p == &*anywhere* otherwise.
	 in fact, for the moment we generate p == &*anywhere* in all cases
      */
      effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
      cell_interpretation rhs_kind = make_cell_interpretation_address_of();
      FOREACH(CELL_RELATION, pv_in, l_in)
	{
	  /* dealing with first cell */
	  /* not generic */
	  effect eff_alias = make_effect(copy_cell(cell_relation_first_cell(pv_in)),
					 make_action_write_memory(),
					 make_approximation_may(),
					 make_descriptor_none());

	  list l_gen_pv = (* ctxt->make_pv_from_effects_func)
	    (eff_alias, anywhere_eff, rhs_kind, l_in);
	  l_gen = (*ctxt->pvs_must_union_func)(l_gen_pv, l_gen);
	  free_effect(eff_alias);

	  if (cell_relation_second_value_of_p(pv_in)
	      && !undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in))
	      && !null_pointer_value_cell_p(cell_relation_second_cell(pv_in)) )
	    {
	      /* not generic */
	      effect eff_alias = make_effect(copy_cell(cell_relation_second_cell(pv_in)),
					     make_action_write_memory(),
					     make_approximation_may(),
					     make_descriptor_none());

	      list l_gen_pv = (* ctxt->make_pv_from_effects_func)
		(eff_alias, anywhere_eff, rhs_kind, l_in);
	      l_gen = (*ctxt->pvs_must_union_func)(l_gen_pv, l_gen);
	      free_effect(eff_alias);
	    }
	}
      free_effect(anywhere_eff);
      free_cell_interpretation(rhs_kind);
    }
  else
    {
      /* generate for all alias p in l_kill p == rhs_eff */
      /* except for p==undefined or p==null (should other abstract values/locations be ignored here?) */
      FOREACH(EFFECT, eff_alias, l_kill)
	{
	  if (!null_pointer_value_effect_p(eff_alias)
	      && ! undefined_pointer_value_effect_p(eff_alias))
	    {
	      list l_rhs_kind_tmp = l_rhs_kind;
	      FOREACH(EFFECT, rhs_eff, l_rhs_eff)
		{
		  cell_interpretation rhs_kind =
		    CELL_INTERPRETATION(CAR(l_rhs_kind_tmp));
		  //bool exact_preceding_p = true;
		  list l_gen_pv = (* ctxt->make_pv_from_effects_func)
		    (eff_alias, rhs_eff, rhs_kind, l_in);
		  l_gen = gen_nconc(l_gen_pv, l_gen);
		  POP(l_rhs_kind_tmp);
		}
	    }
	}
      if (declaration_p)
	{
	  gen_full_free_list(l_kill);
	  l_kill = NIL;
	}
      pips_debug_pvs(2, "l_gen =", l_gen);
    }

  /* now take kills into account */
  l_out = kill_pointer_values(l_in, l_kill, ctxt);
  pips_debug_pvs(2, "l_out_after kill:", l_out);

  /* and add gen */
  l_out = (*ctxt->pvs_must_union_func)(l_out, l_gen);

  pv_res->l_out = l_out;

  return;
}


/*
  @brief returns in pv_res the effects of a multiple pointer assignment (through nested strutures or arrays) on pointer values
  @param lhs_base_eff is the left hand side path of the assignment
  @param l_rhs_base_eff is a list of rhs paths corresponding to the rhs
  @param l_rhs_base_kind is the list of rhs paths interpretations corresponding to elements of l_rhs_eff
  @param l_in is a list of the input pointer values
  @param pv_res is the struture holding the output result
  @param ctxt gives the functions specific to the kind of pointer values to be
          computed.
 */
void multiple_pointer_assignment_to_post_pv(effect lhs_base_eff, type lhs_type,
					    list l_rhs_base_eff, list l_rhs_base_kind,
					    bool declaration_p, list l_in,
					    pv_results *pv_res, pv_context *ctxt)
{
  list l_in_cur = l_in;
  cell rhs_base_cell = gen_length(l_rhs_base_eff) > 0
    ? effect_cell(EFFECT(CAR(l_rhs_base_eff))): cell_undefined;
  bool anywhere_lhs_p = false;

  pips_debug(1, "begin\n");

  pips_assert("assigning NULL to several pointers"
	      " at the same time is forbidden!\n", !(!cell_undefined_p(rhs_base_cell) && null_pointer_value_cell_p(rhs_base_cell)));

  /* we could be more precise here on abstract locations */
  if (anywhere_effect_p(lhs_base_eff))
    {
      pips_assert("we cannot have an anywhere lhs for a declaration\n", !declaration_p);
      pips_debug(3, "anywhere lhs\n");
      anywhere_lhs_p = true;

      list l_rhs_eff = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);
      list l_rhs_kind = CONS(CELL_INTERPRETATION, make_cell_interpretation_address_of(), NIL);

      single_pointer_assignment_to_post_pv(lhs_base_eff, l_rhs_eff, l_rhs_kind, declaration_p, l_in_cur, pv_res, ctxt);

      gen_full_free_list(l_rhs_eff);
      gen_full_free_list(l_rhs_kind);
    }
  else /* if (!anywhere_lhs_p) */
    {
      /* lhs is not a pointer, but it is an array of pointers or an aggregate type
	 with pointers.... */
      /* In this case, it cannot be an address_of case */

      /* First, search for all accessible pointers */
      list l_lhs = generic_effect_generate_all_accessible_paths_effects_with_level
	(lhs_base_eff, lhs_type, is_action_write, false, 0, true);
      if(effect_exact_p(lhs_base_eff))
	effects_to_must_effects(l_lhs); /* to work around the fact that exact effects are must effects */
      pips_debug_effects(2, "l_lhs = ", l_lhs);

      /* Then for each found pointer, do as if there were an assignment by translating
         the rhs path accordingly 
      */
      if (!ENDP(l_lhs))
	{
	  list l_lhs_tmp = l_lhs;
	  while (!anywhere_lhs_p && !ENDP(l_lhs_tmp))
	    {
	      /* build the list of corresponding rhs */
	      effect lhs_eff = EFFECT(CAR(l_lhs_tmp));
	      reference lhs_ref = effect_any_reference(lhs_eff);
	      list lhs_dims = reference_indices(lhs_ref);

	      reference lhs_base_ref = effect_any_reference(lhs_base_eff);
	      size_t lhs_base_nb_dim = gen_length(reference_indices(lhs_base_ref));
	      list l_rhs_eff = NIL;
	      list l_rhs_kind = NIL;

	      bool free_rhs_kind = false;
	      list l_rhs_base_kind_tmp = l_rhs_base_kind;
	      FOREACH(EFFECT, rhs_base_eff, l_rhs_base_eff)
		{
		  effect rhs_eff = copy_effect(rhs_base_eff);
		  cell_interpretation rhs_kind = CELL_INTERPRETATION(CAR(l_rhs_base_kind));

		  if (!undefined_pointer_value_cell_p(rhs_base_cell)
		      && !anywhere_effect_p(rhs_base_eff))
		    {
		      reference rhs_ref = effect_any_reference(rhs_eff);
		      bool to_be_freed = false;
		      type rhs_type = cell_reference_to_type(rhs_ref, &to_be_freed);

		      if (!type_equal_p(lhs_type, rhs_type))
			{
			  pips_debug(5, "not same lhs and rhs types generating anywhere rhs\n");
			  rhs_eff = make_anywhere_effect(make_action_write_memory()); /* should be refined */
			  rhs_kind = make_cell_interpretation_address_of();
			  free_rhs_kind = true;
			}
		      else /* general case at least :-) */
			{
			  /* This is not generic, I should use a translation algorithm here I guess */
			  /* first skip dimensions of kill_ref similar to lhs_base_ref */
			  list lhs_dims_tmp = lhs_dims;
			  for(size_t i = 0; i < lhs_base_nb_dim; i++, POP(lhs_dims_tmp));
			  /* add the remaining dimensions to the copy of rhs_base_eff */
			  FOREACH(EXPRESSION, dim, lhs_dims_tmp)
			    {
			      (*effect_add_expression_dimension_func)(rhs_eff, dim);
			    }
			}
		      if (to_be_freed) free_type(rhs_type);
		    }
		  l_rhs_eff = CONS(EFFECT, rhs_eff, l_rhs_eff);
		  l_rhs_kind = CONS(CELL_INTERPRETATION, rhs_kind, l_rhs_kind);
		  POP(l_rhs_base_kind_tmp);
		}

	      single_pointer_assignment_to_post_pv(lhs_eff, l_rhs_eff, l_rhs_kind, declaration_p, l_in_cur, pv_res, ctxt);

	      gen_full_free_list(l_rhs_eff);
	      if (free_rhs_kind) gen_full_free_list(l_rhs_kind); else gen_free_list(l_rhs_kind);

	      list l_out = pv_res->l_out;
	      if (l_out != l_in_cur)
		gen_full_free_list(l_in_cur);
	      l_in_cur = l_out;

	      list l_result_paths = pv_res->result_paths;

	      if (gen_length(l_result_paths) > 0 && anywhere_effect_p(EFFECT(CAR(l_result_paths))))
		anywhere_lhs_p = true;

	      POP(l_lhs_tmp);
	    } /* while */
	} /* if (!ENDP(l_lhs)) */
      gen_full_free_list(l_lhs);
    } /* if (!anywhere_lhs_p) */

  return;
}


/*
  @brief computes the gen, post and kill pointer values of an assignment
  @param lhs is the left hand side expression of the assignment*
  @param may_lhs_p is true if it's only a possible assignment
  @param rhs is the right hand side of the assignement
  @param l_in is a list of the input pointer values
  @param ctxt gives the functions specific to the kind of pointer values to be
          computed.
 */
void assignment_to_post_pv(expression lhs, bool may_lhs_p,
			   expression rhs, bool declaration_p,
			   list l_in, pv_results *pv_res, pv_context *ctxt)
{
  list l_in_cur = NIL;
  effect lhs_eff = effect_undefined;

  pips_debug(1, "begin with may_lhs_p = %s and declaration_p = %s\n", bool_to_string(may_lhs_p), bool_to_string(declaration_p));
  pips_debug_pvs(2, "input pointer values:\n", l_in);
  type lhs_type = expression_to_type(lhs);

  /* first convert the rhs and lhs into memory paths, rhs is evaluated first */
  /* this is done even if this is a non-pointer assignment, becasue there
     maybe side effects on alising hidden in sub-expressions, function calls...
  */
  pv_results lhs_pv_res = make_pv_results();
  pv_results rhs_pv_res = make_pv_results();

  expression_to_post_pv(rhs, l_in, &rhs_pv_res, ctxt);
  list l_rhs_eff = rhs_pv_res.result_paths;
  list l_rhs_kind = rhs_pv_res.result_paths_interpretations;
  l_in_cur = rhs_pv_res.l_out;

  expression_to_post_pv(lhs, l_in_cur, &lhs_pv_res, ctxt);
  l_in_cur = lhs_pv_res.l_out;
  /* we should test here that lhs_pv_res.result_paths has only one element.
     well is it correct? can a relational operator expression be a lhs ?
  */
  lhs_eff = EFFECT(CAR(lhs_pv_res.result_paths));
  if (may_lhs_p) effect_to_may_effect(lhs_eff);
  pv_res->result_paths = CONS(EFFECT, copy_effect(lhs_eff), NIL);
  pv_res->result_paths_interpretations = CONS(CELL_INTERPRETATION,
					      make_cell_interpretation_value_of(), NIL);

  if (type_fundamental_basic_p(lhs_type) || !basic_concrete_type_leads_to_pointer_p(lhs_type))
    {
      pips_debug(2, "non-pointer assignment\n");
      /* l_gen = NIL; l_kill = NIL; */
      pv_res->l_out = l_in_cur;
    }
  else
    {
      if(type_variable_p(lhs_type))
	{
	  if (pointer_type_p(lhs_type)) /* simple case first: lhs is a pointer */
	    {
	      single_pointer_assignment_to_post_pv(lhs_eff, l_rhs_eff,
						       l_rhs_kind,
						       declaration_p, l_in_cur,
						       pv_res, ctxt);
	    }
	  else /* hidden pointer assignments */
	    {
	      multiple_pointer_assignment_to_post_pv(lhs_eff, lhs_type, l_rhs_eff,
							 l_rhs_kind,
							 declaration_p, l_in_cur,
							 pv_res, ctxt);
	    }
	} /* if (type_variable_p(lhs_type) */
      else if(type_functional_p(lhs_type))
	{
	  pips_internal_error("not yet implemented");
	}
      else
	pips_internal_error("unexpected_type");
    }


  free_pv_results_paths(&lhs_pv_res);
  free_pv_results_paths(&rhs_pv_res);

  pips_debug_pv_results(1, "end with pv_res =\n", *pv_res);
  return;
}

static list module_initial_parameter_pv()
{
  list pv_out = NIL;
  entity module = get_current_module_entity();
  list l_formals = module_formal_parameters(module);
  const char* mod_name = get_current_module_name();
  entity pointer_dummy_area = FindOrCreateEntity(mod_name, POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME);
  storage pointer_dummy_storage = make_storage_ram(make_ram(module, pointer_dummy_area, UNKNOWN_RAM_OFFSET, NIL));

  FOREACH(ENTITY, formal_ent, l_formals)
    {
      effect formal_eff = make_reference_simple_effect(make_reference(formal_ent, NIL),
						       make_action_write_memory(),
						       make_approximation_exact());
      type formal_t = entity_basic_concrete_type(formal_ent);

      // generate all possible effects from the formal parameter
      list lw = generic_effect_generate_all_accessible_paths_effects_with_level(formal_eff,
										formal_t,
										'w',
										true,
										10, /* to avoid too long paths until GAPS are handled */
										false);
      const char * formal_name = entity_user_name(formal_ent);
      int nb = 1;

      FOREACH(EFFECT, eff, lw)
	{
	  // this should be at least partly be embedded in a representation dependent wrapper
	  pips_debug_effect(3, "current effect: \n", eff);
	  bool to_be_freed = false;
	  type eff_t = cell_to_type(effect_cell(eff), &to_be_freed);
	  if (type_variable_p(eff_t) && basic_pointer_p(variable_basic(type_variable(eff_t))))
	    {

	      // we generate an address_of pv, the source of which is the effect cell
	      // and the sink of which is the first element of a new entity, whose type
	      // is a one dimensional array of the pointed type.

	      // first generate the new entity name
	      entity new_ent = entity_undefined;
	      string new_name = NULL;
	      new_name = strdup(concatenate(mod_name, MODULE_SEP_STRING, "_", formal_name,"_", i2a(nb), NULL));
	      nb ++;

	      // then take care of the type and target path indices
	      type pointed_type = basic_pointer(variable_basic(type_variable(eff_t)));
	      type new_type = type_undefined;
	      list new_dims = NIL; // array dimensions of new type

	      // if the effect path contains multiple paths (a[*] or p.end[*].begin for instance)
	      // then there are several targets;
	      // the indices of the new_path must have an additional dimension
	      // and also the new type
	      bool single_path = true;
	      list new_inds = NIL; // indices of the target path

	      // Not generic
	      // for simple cells, isn't it sufficient to test the approximation of the effect ?
	      FOREACH(EXPRESSION, eff_ind_exp, reference_indices(effect_any_reference(eff)))
		{
		  if (unbounded_expression_p(eff_ind_exp))
		    {
		      single_path = false;
		      break;
		    }
		}
	      if (!single_path)
		{
		  new_inds = CONS(EXPRESSION, make_unbounded_expression(), NIL);
		  new_dims = CONS(DIMENSION,
				      make_dimension(int_to_expression(0),
						     make_unbounded_expression()), NIL);
		  // the approximation will be ok because, as the path is not unique
		  // the effect approximation is may by construction
		}

	      if (type_variable_p(pointed_type))
		{

		  pips_debug(5, "variable case\n");
		  variable pointed_type_v = type_variable(pointed_type);
		  basic new_basic = copy_basic(variable_basic(pointed_type_v));
		  if (ENDP(variable_dimensions(pointed_type_v)))
		    {
		      pips_debug(5, "with 0 dimension\n");
		      new_dims = gen_nconc(new_dims, CONS(DIMENSION,
				      make_dimension(int_to_expression(0),
						     make_unbounded_expression()), NIL));
		      new_inds = gen_nconc(new_inds, CONS(EXPRESSION, int_to_expression(0), NIL));
		    }
		  else
		    {
		      new_dims = gen_full_copy_list(variable_dimensions(pointed_type_v));
		      pips_debug(5, "with %d dimension\n", (int) gen_length(new_dims));
		      FOREACH(DIMENSION, dim, new_dims)
			{
			  new_inds = gen_nconc(new_inds, CONS(EXPRESSION, int_to_expression(0), NIL));
			}
		    }
		  new_type = make_type_variable(make_variable(new_basic, new_dims, NIL));
		  pips_debug(5, "new_type is: %s (%s)\n", words_to_string(words_type(new_type, NIL, false)),
			     type_to_string(new_type));
		  new_ent = make_entity(new_name,
					   new_type,
					   copy_storage(pointer_dummy_storage),
					   make_value_unknown());
		}
	      else
		{
		  pips_debug(5, "non-variable formal parameter target type -> anywhere\n");
		  new_ent = entity_all_locations();
		  gen_full_free_list(new_inds);
		  new_inds = NIL;
		}


	      cell new_cell = make_cell_reference(make_reference(new_ent, new_inds));
	      // then make the new pv
	      cell_relation new_pv = make_address_of_pointer_value(copy_cell(effect_cell(eff)),
					    new_cell,
					    effect_approximation_tag(eff),
					    make_descriptor_none());
	      pips_debug_pv(3, "generated pv: \n", new_pv);
	      // and add it to the return list of pvs
	      pv_out = CONS(CELL_RELATION, new_pv, pv_out);
	    }
	  if (to_be_freed) free_type(eff_t);
	}
        gen_full_free_list(lw);
    }
  free_storage(pointer_dummy_storage);
  return pv_out;
}

/**
   @brief generic interface to compute the pointer values of a given module
   @param module_name is the name of the module
   @param ctxt gives the functions specific to the kind of pointer values to be
          computed.
 */
static void generic_module_pointer_values(char * module_name, pv_context *ctxt)
{

  /* temporary settings : in an interprocedural context we need to keep track
     of visited modules */
  /* Get the code of the module. */
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true));
  set_current_module_entity(module_name_to_entity(module_name));
  init_pv();

  debug_on("POINTER_VALUES_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  list l_init = module_initial_parameter_pv();
  if(entity_main_module_p(get_current_module_entity()))
    {
      l_init = (*ctxt->pvs_must_union_func)(l_init, gen_full_copy_list( (*ctxt->db_get_program_pv_func)() ));
    }

  list l_out = statement_to_post_pv(get_current_module_statement(), gen_full_copy_list(l_init), ctxt);

  (*ctxt->db_put_pv_func)(module_name, get_pv());
  (*ctxt->db_put_in_pv_func)(module_name, l_init);
  (*ctxt->db_put_out_pv_func)(module_name, l_out);

  pips_debug(1, "end\n");
  debug_off();
  reset_current_module_entity();
  reset_current_module_statement();

  reset_pv();

  return;
}


static void generic_module_initial_pointer_values(char * module_name, pv_context *ctxt)
{
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true));
  set_current_module_entity(module_name_to_entity(module_name));
  init_pv();

  debug_on("POINTER_VALUES_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  list l_decl = module_to_all_declarations(get_current_module_entity());

  list l_in = NIL;
  list l_cur = l_in;
  FOREACH(ENTITY, decl, l_decl)
    {
      if(variable_static_p(decl))
	{
	  l_cur = declaration_to_post_pv(decl, l_cur, ctxt);
	}
    }
  list l_out = l_cur;

  pips_debug_pvs(1, "Resulting pointer values:\n", l_out);

  (*ctxt->db_put_initial_pv_func)(module_name, l_out);

  pips_debug(1, "end\n");
  debug_off();
  reset_current_module_entity();
  reset_current_module_statement();

  reset_pv();

  return;
}

static void generic_program_pointer_values(char * prog_name, pv_context *ctxt)
{
  entity the_main = get_main_entity();
  list l_out = NIL;
  pips_assert("main was found", the_main!=entity_undefined);

  debug_on("POINTER_VALUES_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  pips_debug(1, "considering program \"%s\" with main \"%s\"\n", prog_name,
	       module_local_name(the_main));
  /* Unavoidable pitfall: initializations in uncalled modules may be
   * taken into account. It all depends on the "create" command.
     */
  gen_array_t modules = db_get_module_list();
  int nmodules = gen_array_nitems(modules);
  pips_assert("some modules in the program", nmodules>0);

  for(int i=0; i<nmodules; i++)
    {
      string module_name = gen_array_item(modules, i);
      pips_debug(1, "considering module %s\n", module_name);

      set_current_module_entity(module_name_to_entity(module_name));
      list l_cur = (ctxt->db_get_initial_pv_func)(module_name);
      pips_debug_pvs(2, "module initial pvs: \n", l_cur);
      reset_current_module_entity();
      l_out = gen_nconc(gen_full_copy_list(l_cur), l_out);
  }

  set_current_module_entity(the_main);
  pips_debug_pvs(2, "storing program pvs: \n", l_out);
  reset_current_module_entity();

  (*ctxt->db_put_program_pv_func)(l_out);
  pips_debug(1, "end\n");
  debug_off();
  return;
}

/**************** INTERFACE *************/

/**
   @brief interface to compute the simple pointer values of a given module
 */
bool simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  set_methods_for_simple_effects();
  generic_module_pointer_values(module_name, &ctxt);
  reset_pv_context(&ctxt);
  generic_effects_reset_all_methods();
  return(true);
}

bool initial_simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  ctxt.initial_pointer_values_p = true;
  set_methods_for_simple_effects();
  generic_module_initial_pointer_values(module_name, &ctxt);
  reset_pv_context(&ctxt);
  generic_effects_reset_all_methods();
  return(true);
}

bool program_simple_pointer_values(char * prog_name)
{
  pv_context ctxt = make_simple_pv_context();
  set_methods_for_simple_effects();
  generic_program_pointer_values(prog_name, &ctxt);
  reset_pv_context(&ctxt);
  generic_effects_reset_all_methods();
  return(true);
}

