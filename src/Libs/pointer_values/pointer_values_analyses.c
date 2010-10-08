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

static statement_cell_relations db_get_simple_pv(char * module_name)
{
  return (statement_cell_relations) db_get_memory_resource(DBR_SIMPLE_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_pv(char * module_name, statement_cell_relations scr)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_POINTER_VALUES, module_name, (char*) scr);
}

static statement_cell_relations db_get_simple_gen_pv(char * module_name)
{
  return (statement_cell_relations) db_get_memory_resource(DBR_SIMPLE_GEN_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_gen_pv(char * module_name, statement_cell_relations scr)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_GEN_POINTER_VALUES, module_name, (char*) scr);
}

static statement_effects db_get_simple_kill_pv(char * module_name)
{
  return (statement_effects) db_get_memory_resource(DBR_SIMPLE_KILL_POINTER_VALUES, module_name, TRUE);
}

static void db_put_simple_kill_pv(char * module_name, statement_effects se)
{
   DB_PUT_MEMORY_RESOURCE(DBR_SIMPLE_KILL_POINTER_VALUES, module_name, (char*) se);
}


/******************** ANALYSIS CONTEXT */


pv_context make_simple_pv_context()
{
  pv_context ctxt;
  
  ctxt.db_get_pv_func = db_get_simple_pv;
  ctxt.db_put_pv_func = db_put_simple_pv;
  ctxt.db_get_gen_pv_func = db_get_simple_gen_pv;
  ctxt.db_put_gen_pv_func = db_put_simple_gen_pv;
  ctxt.db_get_kill_pv_func = db_get_simple_kill_pv;
  ctxt.db_put_kill_pv_func = db_put_simple_kill_pv;
  ctxt.make_pv_from_effects_func = make_simple_pv_from_simple_effects;
  ctxt.cell_reference_with_value_of_cell_reference_translation_func = 
    simple_cell_reference_with_value_of_cell_reference_translation;
  ctxt.cell_reference_with_address_of_cell_reference_translation_func = 
    simple_cell_reference_with_address_of_cell_reference_translation;
  ctxt.pv_composition_with_transformer_func = simple_pv_composition_with_transformer;
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
  p_ctxt->db_get_gen_pv_func =(statement_cell_relations_function) UNDEF ;
  p_ctxt->db_put_gen_pv_func = (void_function) UNDEF;
  p_ctxt->db_get_kill_pv_func = (statement_effects_function) UNDEF;
  p_ctxt->db_put_kill_pv_func = (void_function) UNDEF;
  p_ctxt->make_pv_from_effects_func = (cell_relation_function) UNDEF;
  p_ctxt->pv_composition_with_transformer_func = (cell_relation_function) UNDEF;
}

/***************** ABSTRACT VALUES */

entity undefined_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     UNDEFINED_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
  }
  return u;
}

entity null_pointer_value_entity()
{
  entity u = entity_undefined;
  string u_name = strdup(concatenate(ANY_MODULE_NAME,
				     MODULE_SEP_STRING,
				     NULL_POINTER_VALUE_NAME,
				     NULL));
  u = gen_find_tabulated(u_name, entity_domain);
  if(entity_undefined_p(u)) {
    area a = make_area(0,NIL); /* Size and layout are unknown */
    type t = make_type_area(a);
    u = make_entity(u_name,
		    t, make_storage_rom(), make_value_unknown());
  }
  return u;
}


cell make_undefined_pointer_value_cell()
{
  entity u = undefined_pointer_value_entity();
  return make_cell_reference(make_reference(u, NIL)); 
}

bool undefined_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), UNDEFINED_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;  
}

bool undefined_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else 
    r = preference_reference(cell_preference(c));
  return(undefined_pointer_value_entity_p(reference_variable(r)));
}

cell make_null_pointer_value_cell()
{
  entity u = null_pointer_value_entity();
  return make_cell_reference(make_reference(u, NIL));
}

bool null_pointer_value_entity_p(entity e)
{
  bool res;
  res = same_string_p(entity_local_name(e), NULL_POINTER_VALUE_NAME);
  res = res && same_string_p(entity_module_name(e), ANY_MODULE_NAME);
  return res;  
}

bool null_pointer_value_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else 
    r = preference_reference(cell_preference(c));
  return(null_pointer_value_entity_p(reference_variable(r)));
}


bool anywhere_cell_p(cell c)
{
  reference r;
  if (cell_gap_p(c)) return false;
  else if (cell_reference_p(c))
    r = cell_reference(c);
  else 
    r = preference_reference(cell_preference(c));
  return(entity_all_locations_p(reference_variable(r)));  
}

/******************** CONVERSION FROM EXPRESSIONS TO PATHS AND CELL_INTERPRETATION */

static effect intrinsic_to_interpreted_path(entity func, list args, cell_interpretation * ci, pv_context * ctxt);
static effect call_to_interpreted_path(call c, cell_interpretation * ci, pv_context * ctxt);
static 
effect expression_to_interpreted_path(expression exp, cell_interpretation * ci, pv_context * ctxt);


static effect intrinsic_to_interpreted_path(entity func, list args, cell_interpretation * ci, pv_context * ctxt)
{
  effect eff = effect_undefined;
  
  if(ENTITY_ASSIGN_P(func))
    /* if the call is an assignement, return the lhs path because the relation
       between the lhs and the rhs has been treated elsewhere
    */
    return(expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt));

  if(ENTITY_DEREFERENCING_P(func))
    {
      eff = expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt);
      effect_add_dereferencing_dimension(eff);
      *ci = make_cell_interpretation_value_of();
      return eff;
    }

  if(ENTITY_FIELD_P(func))
    {
      expression e2 = EXPRESSION(CAR(CDR(args)));
      syntax s2 = expression_syntax(e2);
      reference r2 = syntax_reference(s2);
      entity f = reference_variable(r2);

      pips_assert("e2 is a reference", syntax_reference_p(s2));
      pips_debug(4, "It's a field operator\n");

      eff = expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt);
      effect_add_field_dimension(eff,f);
      *ci = make_cell_interpretation_value_of();
      return eff;
    }

  if(ENTITY_POINT_TO_P(func))
    {
      expression e2 = EXPRESSION(CAR(CDR(args)));
      syntax s2 = expression_syntax(e2);
      entity f;

      pips_assert("e2 is a reference", syntax_reference_p(s2));
      f = reference_variable(syntax_reference(s2));

      pips_debug(4, "It's a point to operator\n");
      eff = expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt);

      /* We add a dereferencing */
      effect_add_dereferencing_dimension(eff);

      /* we add the field dimension */
      effect_add_field_dimension(eff,f);
      *ci = make_cell_interpretation_value_of();
      return eff;
    }

  if(ENTITY_ADDRESS_OF_P(func))
    {
      eff = expression_to_interpreted_path(EXPRESSION(CAR(args)), ci, ctxt);
      *ci = make_cell_interpretation_address_of();
      return eff;      
    }
  
  return eff;
}

static effect call_to_interpreted_path(call c, cell_interpretation * ci, pv_context * ctxt)
{
  effect eff = effect_undefined;

  entity func = call_function(c);
  value func_init = entity_initial(func);
  tag t = value_tag(func_init);
  string n = module_local_name(func);
  list func_args = call_arguments(c);
  type uet = ultimate_type(entity_type(func));
  
  if(type_functional_p(uet)) 
    {
      switch (t) 
	{
	case is_value_code:
	  pips_debug(5, "external function %s\n", n);
	  /* If it's an external call, we should try to retrieve the returned value 
	     if it's a pointer. 
	     for the moment, just return an anywhere effect....
	  */
	  pips_user_warning("external call, not handled yet, returning all locations effect\n");
	  eff = make_anywhere_effect(make_action_write_memory());
	  *ci = make_cell_interpretation_address_of();		  
	  break;
      
	case is_value_intrinsic:
	  pips_debug(5, "intrinsic function %s\n", n);
	  eff = intrinsic_to_interpreted_path(func, func_args, ci, ctxt);
	  break;
      
	case is_value_symbolic:
	  pips_debug(5, "symbolic\n");
	  pips_internal_error("symbolic case, not yet implemented\n", entity_name(func));
	  break;
	    
	case is_value_constant:
	  pips_debug(5, "constant\n");
	  constant func_const = value_constant(func_init);
	  /* We should be here only in case of a pointer value rhs, and the value should be 0 */
	  if (constant_int_p(func_const))
	    {
	      if (constant_int(func_const) == 0)
		{
		  *ci = make_cell_interpretation_value_of();
		  cell path = make_null_pointer_value_cell();
		  /* use approximation_must to be consistent with effects, should be approximation_exact */
		  eff = make_effect(path, make_action_read_memory(), make_approximation_must(), make_descriptor_none());
		}
	      else
		pips_internal_error("unexpected integer constant value\n", entity_name(func));
	    }
	  else
	    pips_internal_error("unexpected constant case\n", entity_name(func));
	  break;
	    
	case is_value_unknown:
	  pips_internal_error("unknown function %s\n", entity_name(func));
	    
	default:
	  pips_internal_error("unknown tag %d\n", t);
	  break;
	}
    }
  return eff;
}

static 
effect expression_to_interpreted_path(expression exp, cell_interpretation * ci, 
				      pv_context * ctxt)
{
  effect eff = effect_undefined;
  *ci = cell_interpretation_undefined;

  if (expression_undefined_p(exp))
    {
      eff = make_effect(make_undefined_pointer_value_cell(), 
			make_action_write_memory(), 
			make_approximation_must(), 
			make_descriptor_none());  
      *ci = make_cell_interpretation_value_of();
    }
  else
    {
      pips_debug(1, "begin with_expression : %s\n", 
		 words_to_string(words_expression(exp,NIL)));

      syntax exp_syntax = expression_syntax(exp);

      switch(syntax_tag(exp_syntax))
	{
	case is_syntax_reference:
	  pips_debug(5, "reference case\n");
	  reference exp_ref = syntax_reference(exp_syntax);
	  if (same_string_p(entity_local_name(reference_variable(exp_ref)), "NULL"))
	    {
	      *ci = make_cell_interpretation_value_of();
	      cell path = make_null_pointer_value_cell();
	      /* use approximation_must to be consistent with effects, should be approximation_exact */
	      eff = make_effect(path, make_action_read_memory(), make_approximation_must(), make_descriptor_none());
	    }
	  else
	    {
	      /* this function name should be stored in ctxt*/
	      eff = (*reference_to_effect_func)(copy_reference(exp_ref), 
						make_action_write_memory(), false);
	      *ci = make_cell_interpretation_value_of();
	    }
	  break;

	case is_syntax_range:
	  pips_debug(5, "range case\n");
	  pips_internal_error("not yet implemented\n");
	  break;

	case is_syntax_call:
	  pips_debug(5, "call case\n");
	  eff = call_to_interpreted_path(syntax_call(exp_syntax), ci, ctxt);
	  break;

	case is_syntax_cast:
	  pips_debug(5, "cast case\n");
	  eff = expression_to_interpreted_path(cast_expression(syntax_cast(exp_syntax)), 
					       ci, ctxt);
	  break;

	case is_syntax_sizeofexpression:
	  pips_debug(5, "sizeof case\n");
	  pips_internal_error("sizeof not expected\n");
	  break;
    
	case is_syntax_subscript:
	  pips_debug(5, "subscript case\n");
	  subscript sub = syntax_subscript(exp_syntax);
	  eff = expression_to_interpreted_path(subscript_array(sub), ci, ctxt);

	  FOREACH(EXPRESSION, sub_ind_exp, subscript_indices(sub))
	    {
	      (*effect_add_expression_dimension_func)(eff, sub_ind_exp);
	    }
	  list l_tmp = generic_proper_effects_of_complex_address_expression(exp, &eff, true);
	  *ci = make_cell_interpretation_value_of();
	  gen_full_free_list(l_tmp);	
	  break;

	case is_syntax_application:
	  pips_debug(5, "application case\n");		
	  pips_internal_error("not yet implemented\n");
	  break;

	case is_syntax_va_arg:
	  pips_debug(5, "va_arg case\n");		
	  pips_internal_error("va_arg not expected\n");
	  break;
      
	default:
	  pips_internal_error("unexpected tag %d\n", syntax_tag(exp_syntax));
	}
    }
  pips_debug_effect(2, "returning path :",eff);
  pips_debug(2, "with %s interpretation\n", 
	     cell_interpretation_value_of_p(*ci)? "value_of": "address_of");
  pips_debug(1,"end\n");
  return eff;
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
list expression_to_post_pv(expression exp, list l_in, pv_context *ctxt);

static
list call_to_post_pv(call c, list l_in, pv_context *ctxt);

static
list intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_context *ctxt);

static 
list assignment_to_post_pv(expression lhs, expression rhs, list l_in, pv_context *ctxt);



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

	pips_debug(2, "dealing with statement ");
	print_statement(stmt);
	pips_debug_pvs(2, "l_cur = \n", l_cur);
      }
      /* keep local variables in declaration reverse order */
      if (declaration_statement_p(stmt))
	{
	  store_pv(stmt, make_cell_relations(gen_full_copy_list(l_cur)));
	  FOREACH(ENTITY, e, statement_declarations(stmt))
	    {
	      /* beware don't push static variables */
	      l_locals = CONS(ENTITY, e, l_locals);
	      l_cur = declaration_to_post_pv(e, l_cur, ctxt);
	    }
	  store_gen_pv(stmt, make_cell_relations(NIL));
	  store_kill_pv(stmt, make_effects(NIL));
	  
	} 
      else
	l_cur = statement_to_post_pv(stmt, l_cur, ctxt);

    }
  
  /* don't forget to eliminate local declarations on exit */
  /* ... */
  pips_debug_pvs(2, "returning: ", l_cur);
  pips_debug(1, "end\n");
  return (l_cur);
}

static 
list statement_to_post_pv(statement stmt, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_debug_pvs(2, "input pvs: ", l_in);

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

  pips_debug_pvs(2, "before composition_with_transformer: ", l_out);
  l_out = pvs_composition_with_transformer(l_out, transformer_undefined, ctxt);

  store_gen_pv(stmt, make_cell_relations(NIL));
  store_kill_pv(stmt, make_effects(NIL));
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");

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
  pips_debug(1, "begin\n");

  expression lhs_exp = entity_to_expression(e);
  expression rhs_exp = expression_undefined;
  bool free_rhs_exp = false;

  value v_init = entity_initial(e);
  if (v_init == value_undefined)
    {
      pips_debug(2, "undefined inital value\n");
      rhs_exp = entity_to_expression(undefined_pointer_value_entity());
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
	  pips_internal_error("unexpected tag\n");
	}
    }

  l_out = assignment_to_post_pv(lhs_exp, rhs_exp, l_in, ctxt);
  
  free_expression(lhs_exp);
  if (free_rhs_exp) free_expression(rhs_exp);
  


  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
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
      l_out = expression_to_post_pv(instruction_expression(inst), l_in, ctxt);
      break;
    case is_instruction_call:
      l_out = call_to_post_pv(instruction_call(inst), l_in, ctxt);
      break;
    case is_instruction_goto:
      pips_internal_error("unexpected goto in pointer values analyses\n");
      break;
    case is_instruction_multitest:
      pips_internal_error("unexpected multitest in pointer values analyses\n");
      break;
    default:
      pips_internal_error("unknown instruction tag\n");
    }
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  
  return (l_out);
}

static
list test_to_post_pv(test t, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list loop_to_post_pv(loop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug(1, "end\n");
  return (l_out);
}

static
list whileloop_to_post_pv(whileloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list forloop_to_post_pv(forloop l, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  pips_internal_error("not yet implemented\n");
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}


static
list unstructured_to_post_pv(unstructured u, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_internal_error("not yet implemented\n");
   pips_debug_pvs(2, "returning: ", l_out);
 pips_debug(1, "end\n");
  return (l_out);
}

static
list expression_to_post_pv(expression exp, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin\n");
  syntax s = expression_syntax(exp);

  switch(syntax_tag(s))
    {
    case is_syntax_reference:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_range:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_call:
      {
	l_out = call_to_post_pv(syntax_call(s), l_in, ctxt);
	break;
      }
    case is_syntax_cast:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_sizeofexpression:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_subscript:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    case is_syntax_application:
      pips_internal_error("not yet implemented\n");
      break;
    case is_syntax_va_arg:
      {
	pips_internal_error("not yet implemented\n");
	break;
      }
    default:
      pips_internal_error("unexpected tag %d\n", syntax_tag(s));
    }

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list call_to_post_pv(call c, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  entity func = call_function(c);
  tag t = value_tag(entity_initial(func));
  list func_args = call_arguments(c);
  type func_type = ultimate_type(entity_type(func));

  pips_debug(1, "begin for %s\n", entity_local_name(func));

  if(type_functional_p(func_type)) 
    {
      switch (t) 
	{
	case is_value_code:
	  pips_debug(5, "external function \n");
	  /* the expression that denotes the called function and the arguments
	     are evaluated; then there is a sequence point, and the function is called
	  */
	  
	  pips_internal_error("not yet implemented\n");
	  break;
	  
	case is_value_intrinsic:
	  pips_debug(5, "intrinsic function\n");
	  l_out = intrinsic_to_post_pv(func, func_args, l_in, ctxt);
	  break;
	  
	case is_value_symbolic:
	  pips_debug(5, "symbolic\n");
	  l_out = l_in;
	  break;
	  
	case is_value_constant:
	  pips_debug(5, "constant\n");
	  l_out = l_in;
	  break;
	  
	case is_value_unknown:
	  pips_internal_error("unknown function \n");
	  break;
	  
	default:
	  pips_internal_error("unknown tag %d\n", t);
	  break;
	}
    }
  else if(type_variable_p(func_type)) 
    {
      pips_internal_error("not yet implemented\n");
    }
  else if(type_statement_p(func_type)) 
    {
      pips_internal_error("not yet implemented\n");
    }
  else 
    {
      pips_internal_error("Unexpected case\n");
    }

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}

static
list intrinsic_to_post_pv(entity func, list func_args, list l_in, pv_context *ctxt)
{
  list l_out = NIL;
  pips_debug(1, "begin for %s\n", entity_local_name(func));

  /* only few intrinsics are currently handled : we should have a way to 
     describe the effects on aliasing of all intrinsics.
  */

  if (ENTITY_ASSIGN_P(func)) 
    {
      expression lhs = EXPRESSION(CAR(func_args));
      expression rhs = EXPRESSION(CAR(CDR(func_args)));
      l_out = assignment_to_post_pv(lhs, rhs, l_in, ctxt);
    }
  else if((ENTITY_STOP_P(func) || ENTITY_ABORT_SYSTEM_P(func)
	   || ENTITY_EXIT_SYSTEM_P(func))) 
    {
      /* The call is never returned from. No information is available
	 for the dead code that follows.
      */
      l_out = NIL;
    }
  else if (ENTITY_C_RETURN_P(func))
    {
      /* but we have to evaluate the impact
	 of the argument evaluation on pointer values
	 eliminate local variables, retrieve the value of the returned pointer if any...
      */
      l_out = expression_to_post_pv(EXPRESSION(CAR(func_args)), l_in, ctxt);
    }
  else
    pips_internal_error("not yet implemented\n");

  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
}



/*
  @brief computes the gen, post and kill pointer values of an assignment 
  @param lhs is the left hand side expression of the assignment
  @param rhs is the right hand side of the assignement
  @param l_in is a list of the input pointer values
  @param ctxt gives the functions specific to the kind of pointer values to be
          computed. 
 */
static 
list assignment_to_post_pv(expression lhs, expression rhs, list l_in, pv_context *ctxt) 
{
  list l_out = NIL;
  list l_aliased = NIL;
  list l_kill = NIL;
  list l_gen = NIL;
  effect lhs_eff = effect_undefined;
  cell_interpretation lhs_kind;
  effect rhs_eff = effect_undefined;
  cell_interpretation rhs_kind;
  pips_debug(1, "begin\n");

  type lhs_type = expression_to_type(lhs);

  if (type_fundamental_basic_p(lhs_type) || !type_leads_to_pointer_p(lhs_type))
    {
      pips_debug(2, "non-pointer assignment \n");
    }
  else
    {
		    

      if(type_variable_p(lhs_type))
	{
	  bool anywhere_lhs_p = false;

	  /* first convert the rhs and lhs into memory paths */
	  rhs_eff = expression_to_interpreted_path(rhs, &rhs_kind, ctxt);
	  lhs_eff = expression_to_interpreted_path(lhs, &lhs_kind, ctxt);
	  free_cell_interpretation(lhs_kind);
      
	  /* we could be more precise here on abstract locations */
	  if (anywhere_effect_p(lhs_eff))
	    {
	      pips_debug(3, "anywhere lhs\n");
	      anywhere_lhs_p = true;
	      l_kill = CONS(EFFECT, copy_effect(lhs_eff), NIL);
	    }
	  
	  /* find all pointers that are or may be defined through this assignment */
	  if (pointer_type_p(lhs_type))
	    {
	      /* simple case first: lhs is a pointer */
	      if (!anywhere_lhs_p)
		{
		  l_aliased = effect_find_aliased_paths_with_pointer_values(lhs_eff, l_in, ctxt);
		  if (!ENDP(l_aliased) && anywhere_effect_p(EFFECT(CAR(l_aliased))))
		    {
		      pips_debug(3, "anywhere lhs (from aliases)\n");
		      anywhere_lhs_p = true;
		      l_kill = l_aliased;
		    }
		  else
		    {
		      /* if lhs_eff is a may-be-killed, then all aliased effects are also 
			 may-be-killed effects */
		      if (effect_may_p(lhs_eff)) effects_to_may_effects(l_aliased);
		      l_kill = CONS(EFFECT, lhs_eff, l_aliased);
		      pips_debug_effects(2, "pointer case, l_kill = ", l_kill);		      
		    }
		}

	      if (anywhere_lhs_p)
		{
		  /* we must find in l_in all pointers p and generate p == rhs */
		  FOREACH(CELL_RELATION, pv_in, l_in)
		    {
		      if (cell_relation_second_address_of_p(pv_in) 
			  || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in))
			  || null_pointer_value_cell_p(cell_relation_second_cell(pv_in)) )
			{
			  /* not generic */
			  effect eff_alias = make_effect(copy_cell(cell_relation_first_cell(pv_in)),
							 make_action_write_memory(),
							 make_approximation_may(),
							 make_descriptor_none());
			  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
			    (eff_alias, rhs_eff, rhs_kind);
			  l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
			  free_effect(eff_alias);
			}
		    }
		}
	      else
		{
		  FOREACH(EFFECT, eff_alias, l_kill)
		    {
		      cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
			(eff_alias, rhs_eff, rhs_kind);
		      l_gen = CONS(CELL_RELATION, gen_pv, l_gen);	      
		    }
		  pips_debug_pvs(2, "l_gen = ", l_gen);
		  free_effect(rhs_eff);
		  free_cell_interpretation(rhs_kind);
		}
	      
	    }
	  else 
	    {
	      if (!anywhere_lhs_p)
		{
		  /* lhs is not a pointer, but it is an array of pointers or an aggregate type 
		     with pointers.... */
		  /* In this case, it cannot be an address_of case */
		  list l_lhs = generic_effect_generate_all_accessible_paths_effects_with_level
		    (lhs_eff, lhs_type, is_action_write, false, 0, true);
		  if(effect_must_p(lhs_eff))
		    effects_to_must_effects(l_lhs);
		  pips_debug_effects(2, "l_lhs = ", l_lhs);
		  
		  if (!ENDP(l_lhs))
		    {	      
		      pips_assert("pointer assignement through arrays or aggregate types"
				  " requires a value_of rhs kind\n", 
				  cell_interpretation_value_of_p(rhs_kind));
		      cell rhs_cell = effect_cell(rhs_eff);
		      
		      if(null_pointer_value_cell_p(rhs_cell))
			{
			  pips_internal_error("assigning NULL to several pointers"
					      " at the same time!\n");
			}
		      else if (undefined_pointer_value_cell_p(rhs_cell) || 
			       (anywhere_effect_p(rhs_eff)))
			{
			  list l_lhs_tmp = l_lhs;
			  while (!anywhere_lhs_p && !ENDP(l_lhs_tmp))
			    {
			      effect eff = EFFECT(CAR(l_lhs_tmp));
			      l_aliased = effect_find_aliased_paths_with_pointer_values(eff, l_in, ctxt);
			      if (!ENDP(l_aliased) && anywhere_effect_p(EFFECT(CAR(l_aliased))))
				{
				  pips_debug(3, "anywhere lhs (from aliases)\n");
				  anywhere_lhs_p = true;
				  gen_full_free_list(l_kill);
				  l_kill = l_aliased;
				  gen_full_free_list(l_gen);
				  /* we must find in l_in all pointers p and generate p == rhs */
				  effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
				  cell_interpretation ci = make_cell_interpretation_address_of();
				  FOREACH(CELL_RELATION, pv_in, l_in)
				    {
				      if (cell_relation_second_address_of_p(pv_in) 
					  || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in))
					  || null_pointer_value_cell_p(cell_relation_second_cell(pv_in)) )
					{
					  /* not generic */
					  effect eff_alias = make_effect
					    (copy_cell(cell_relation_first_cell(pv_in)),
					     make_action_write_memory(),
					     make_approximation_may(),
					     make_descriptor_none());
					  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
					    (eff_alias, anywhere_eff, ci);
					  l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
					  free_effect(eff_alias);
					}
				    }
				  free_effect(anywhere_eff);
				  free_cell_interpretation(ci);
				}
			      else
				{
				  /* if eff is a may-be-killed, then all aliased effects are also 
				     may-be-killed effects */
				  if (effect_may_p(eff))
				    effects_to_may_effects(l_aliased);
				  l_aliased = CONS(EFFECT, eff, l_aliased);
				  l_kill = gen_nconc(l_kill, l_aliased);
				  pips_debug_effects(2, "pointer case, l_aliased = ", l_aliased);
				  FOREACH(EFFECT, eff_alias, l_aliased)
				    {
				      cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
					(eff_alias, rhs_eff, rhs_kind);
				      l_gen = CONS(CELL_RELATION, gen_pv, l_gen);	      
				    }
				}
			      POP(l_lhs_tmp);
			    } /* while */
			}
		      else
			{
			  reference rhs_ref = effect_any_reference(rhs_eff);
			  type rhs_type = cell_reference_to_type(rhs_ref);
			  
			  if (type_equal_p(lhs_type, rhs_type))
			    {
			      reference lhs_ref = effect_any_reference(lhs_eff);
			      size_t lhs_nb_dim = gen_length(reference_indices(lhs_ref));
			      
			      list l_lhs_tmp = l_lhs;
			      while (!anywhere_lhs_p && !ENDP(l_lhs_tmp))
				{
				  effect eff = EFFECT(CAR(l_lhs_tmp));
				  reference ref = effect_any_reference(eff);
				  list dims = reference_indices(ref);
				  effect new_rhs_eff = copy_effect(rhs_eff);
				  
				  /* first skip dimensions of kill_ref similar to lhs_ref */
				  
				  for(size_t i = 0; i < lhs_nb_dim; i++, POP(dims));
				  
				  /* add the remaining dimensions to the copy of rhs_eff */
				  
				  for(; !ENDP(dims); POP(dims))
				    {
				      expression dim = EXPRESSION(CAR(dims));
				      (*effect_add_expression_dimension_func)(new_rhs_eff, dim);
				      
				    }
				  
				  /* find aliases */
				  l_aliased = effect_find_aliased_paths_with_pointer_values(eff, l_in, ctxt);
				  if (!ENDP(l_aliased) && anywhere_effect_p(EFFECT(CAR(l_aliased))))
				    {
				      pips_debug(3, "anywhere lhs (from aliases)\n");
				      anywhere_lhs_p = true;
				      gen_full_free_list(l_kill);
				      l_kill = l_aliased;
				      gen_full_free_list(l_gen);
				      /* we must find in l_in all pointers p and generate p == rhs */
				      effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
				      cell_interpretation ci = make_cell_interpretation_address_of();
				      FOREACH(CELL_RELATION, pv_in, l_in)
					{
					  if (cell_relation_second_address_of_p(pv_in) 
					      || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in))
					      || null_pointer_value_cell_p(cell_relation_second_cell(pv_in)) )
					    {
					      /* not generic */
					      effect eff_alias = make_effect
						(copy_cell(cell_relation_first_cell(pv_in)),
						 make_action_write_memory(),
						 make_approximation_may(),
						 make_descriptor_none());
					      cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
						(eff_alias, anywhere_eff, ci);
					      l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
					      free_effect(eff_alias);
					    }
					}
				      free_effect(anywhere_eff);
				      free_cell_interpretation(ci);
				    }
				  else
				    {
				      /* if eff is a may-be-killed, then all aliased effects 
					 are also may-be-killed effects */
				      if (effect_may_p(eff))
					effects_to_may_effects(l_aliased);
				      l_aliased = CONS(EFFECT, eff, l_aliased);
				      l_kill = gen_nconc(l_kill, l_aliased);
				      pips_debug_effects(2, "pointer case, l_aliased = ", l_aliased);
				      
				      /* Then build the pointer_value relation */
				      FOREACH(EFFECT, eff_alias, l_aliased)
					{
					  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
					    (eff_alias, new_rhs_eff, rhs_kind);
					  l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
					}
				      free_effect(new_rhs_eff);
				    }
				  POP(l_lhs_tmp);
				} /* while */
			    }
			  else
			    {
			      pips_internal_error("not same lhs and rhs types, not yet supported."
						  " Please report.\n");
			    }
			  
			  free_type(rhs_type);
			}
		      
		      free_effect(rhs_eff); 
		      free_cell_interpretation(rhs_kind);

		    } /* if (!ENDP(l_lhs)) */
		}
	      else
		{
		  /* we must find in l_in all pointers p and generate p == &*anywhere* */
		  effect anywhere_eff = make_anywhere_effect(make_action_write_memory());
		  cell_interpretation ci = make_cell_interpretation_address_of();
		  FOREACH(CELL_RELATION, pv_in, l_in)
		    {
		      if (cell_relation_second_address_of_p(pv_in) 
			  || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in))
			  || null_pointer_value_cell_p(cell_relation_second_cell(pv_in)) )
			{
			  /* not generic */
			  effect eff_alias = make_effect(copy_cell(cell_relation_first_cell(pv_in)),
							 make_action_write_memory(),
							 make_approximation_may(),
							 make_descriptor_none());
			  cell_relation gen_pv = (* ctxt->make_pv_from_effects_func)
			    (eff_alias, anywhere_eff, ci);
			  l_gen = CONS(CELL_RELATION, gen_pv, l_gen);
			  free_effect(eff_alias);
			}
		      free_effect(anywhere_eff);
		    }
		}	
	    }

	  
	  
	} /* if (type_variable_p(lhs_type) */
      else if(type_functional_p(lhs_type))
	{
	  pips_internal_error("not yet implemented\n");
	}
      else
	pips_internal_error("unexpected_type\n");
    }

  /* now take kills into account */
  l_out = kill_pointer_values(l_in, l_kill, ctxt);
  pips_debug_pvs(2, "l_out_after kill: ", l_out);

  /* and add gen */
  list l_tmp = pvs_must_union(l_out, l_gen);
  gen_full_free_list(l_out);
  gen_full_free_list(l_gen);
  l_out = l_tmp;
  free_type(lhs_type);
  pips_debug_pvs(2, "returning: ", l_out);
  pips_debug(1, "end\n");
  return (l_out);
  
}

/* 
   @brief generic interface to compute the pointer values of a given module
   @param module_name is the name of the module
   @param ctxt gives the functions specific to the kind of pointer values to be
          computed.
 */


static void generic_module_pointer_values(char * module_name, pv_context *ctxt)
{
  list l_out;

  /* temporary settings : in an interprocedural context we need to keep track 
     of visited modules */
  /* Get the code of the module. */
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, TRUE));
  
  set_current_module_entity(module_name_to_entity(module_name));
  init_pv();
  init_gen_pv();
  init_kill_pv();

  debug_on("POINTER_VALUES_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  
  l_out = statement_to_post_pv(get_current_module_statement(), NIL, ctxt);

  (*ctxt->db_put_pv_func)(module_name, get_pv());
  (*ctxt->db_put_gen_pv_func)(module_name, get_gen_pv());
  (*ctxt->db_put_kill_pv_func)(module_name, get_kill_pv());

  pips_debug(1, "end\n");
  debug_off();
  reset_current_module_entity();
  reset_current_module_statement();

  reset_pv();
  reset_gen_pv();
  reset_kill_pv();

  return;
}

/**************** INTERFACE *************/

/* 
   @brief interface to compute the simple pointer values of a given module
 */
bool simple_pointer_values(char * module_name)
{
  pv_context ctxt = make_simple_pv_context();
  set_methods_for_simple_pointer_effects();
  generic_module_pointer_values(module_name, &ctxt);
  reset_pv_context(&ctxt);
  generic_effects_reset_all_methods();
  return(TRUE);
}

/**************** UTILS : should be moved elsewhere */

/*
 @brief makes a pointer value cell_relation from the two input effects; the interpretation 
        of the second effect cell is given by ci. The cells and potential descriptors are copied
        so that no sharing is introduced.
 @param lhs_eff effect gives the first cell of the returned pointer value cell_relation
 @param rhs_eff effect gives the second cell of the returned pointer value cell_relation
 @param ci gives the interpretation of rhs_eff cell in the returned cell_relation (either value_of or address_of).

 @return a cell_relation representing a pointer value relation.
 */
cell_relation make_simple_pv_from_simple_effects(effect lhs_eff, effect rhs_eff, cell_interpretation ci)
{
  cell_relation pv;  

  pips_debug(1,"begin for %s cell_interpretation and effects :\n", cell_interpretation_value_of_p(ci)? "value_of" : "address_of");
  pips_debug_effect(2, "lhs_eff: ", lhs_eff);
  pips_debug_effect(2, "rhs_eff: ", rhs_eff);

  tag lhs_t = effect_approximation_tag(lhs_eff); 
  tag rhs_t = effect_approximation_tag(rhs_eff);
  tag t = approximation_and(lhs_t, rhs_t);

  pips_debug(5,"approximation before converting to store independent cells: %s\n",
	     t == is_approximation_must ? "must": "may");   

  if (t == is_approximation_must) t = is_approximation_exact;
  
  cell lhs_c = effect_cell(lhs_eff);
  lhs_c = make_cell(is_cell_reference, copy_reference(effect_any_reference(lhs_eff)));


  cell rhs_c = effect_cell(rhs_eff);
  bool changed_rhs_p = false;
  bool changed_lhs_p = false;

  if (undefined_pointer_value_cell_p(rhs_c))
    rhs_c = make_undefined_pointer_value_cell();
  else
    {
      rhs_c = make_cell(is_cell_reference, copy_reference(effect_any_reference(rhs_eff)));
      
      bool exact_preceding_p = false;
      
      if (simple_cell_reference_preceding_p(cell_reference(lhs_c), descriptor_undefined, 
					    cell_reference(rhs_c), descriptor_undefined, 
					    transformer_undefined,
					    & exact_preceding_p))
	{
	  pips_user_warning("recursive data structure : GAPs not handled yet\n");
	  free_cell(rhs_c);
	  rhs_c = make_cell(is_cell_reference, make_reference(entity_all_locations(), NIL));
	  cell_interpretation_tag(ci) = is_cell_interpretation_address_of;
	  t = is_approximation_may;
	}
      else
	{
	  lhs_c =  simple_cell_to_store_independent_cell(lhs_c, &changed_lhs_p);
	  rhs_c =  simple_cell_to_store_independent_cell(rhs_c, &changed_rhs_p);
	}
    }

  if (changed_lhs_p || changed_rhs_p)
    {
      pips_debug(5, "approximation set to may after change to store independent cell\n");
      t = is_approximation_may;
    }

  if (cell_interpretation_value_of_p(ci))
    pv = make_value_of_pointer_value(lhs_c,
				     rhs_c,
				     t,
				     make_descriptor_none());	
  else
    pv = make_address_of_pointer_value(lhs_c,
				       rhs_c,
				       t,
				       make_descriptor_none());	
  
  pips_debug_pv(2, "generating: ", pv); 
  pips_debug(1,"end\n");
  return pv;
}


list kill_pointer_values(list /* of cell_relations */ l_in, 
			 list /* of effects */ l_kill,
			 pv_context * ctxt)
{
  list l_cur = l_in;
  pips_debug_pvs(5, "l_in = ", l_in);
  pips_debug_effects(5, "l_kill = ", l_kill);
  
  FOREACH(EFFECT, eff_kill, l_kill)
    {
      l_cur = kill_pointer_value(eff_kill, l_cur, ctxt); 
    }

  pips_debug_pvs(5, "returning : ", l_cur);
  return l_cur;
}


/* Not yet very generic: either should be made generic or a specific version made
   for convex pointer values/effects.  */
list kill_pointer_value(effect eff_kill, list /* of cell_relations */ l_in, 
			pv_context * ctxt)
{
  list l_out = NIL;
  


  pips_debug_pvs(1, "begin with l_in: \n", l_in);
  pips_debug_effect(1, "and eff_kill:\n", eff_kill);


  if (anywhere_effect_p(eff_kill))
    {
      /* all pointers may be killed */
      pips_debug(5, "anywhere case \n");

      FOREACH(CELL_RELATION, pv_in, l_in)
	{
	  cell_relation pv_out = copy_cell_relation(pv_in);
	  cell_relation_approximation_tag(pv_out) = is_approximation_may;
	  l_out = CONS(CELL_RELATION, pv_out, l_out);
	}
      
    }
  else
    {
      /* eff_kill characteristics */
      cell cell_kill = effect_cell(eff_kill);
      tag app_kill = effect_approximation_tag(eff_kill);
      reference ref_kill = effect_any_reference(eff_kill);
      entity e_kill = reference_variable(ref_kill);
      list ind_kill = reference_indices(ref_kill);
      size_t nb_ind_kill = gen_length(ind_kill);
      /******/
			
      /* using old_values, take into account the impact of eff_kill on 
	 l_in pointer values second cells which must be expressed in terms of 
	 unchanged paths. 
      */
      list l_remnants = NIL;
      cell_relation exact_old_pv = cell_relation_undefined;
      list l_old_values = NIL;
  
      pips_debug(4, "begin, looking for an exact old value for eff_orig\n");
  
      l_old_values = effect_find_equivalent_pointer_values(eff_kill, l_in, 
							   &exact_old_pv, 
							   &l_remnants);
      pips_debug_pvs(3, "l_old_values: \n", l_old_values);
      pips_debug_pvs(3, "l_remnants: \n", l_remnants);
      pips_debug_pv(3, "exact_old_pv: \n", exact_old_pv);
  
      list l_keep = NIL;
      FOREACH(CELL_RELATION, pv_in, l_remnants)
	{
	  bool first_p = false; /* should we translate the first or the second pv_in cell */
	  bool to_be_translated = false; /* must it be translated ? */
	  bool exact_preceding_test = true;

	  /* pv_in first cell characteristics */     
	  cell cell_in_1 = cell_relation_first_cell(pv_in);
	  reference ref_in_1 = cell_reference(cell_in_1);
	  entity e_in_1 = reference_variable(ref_in_1);
	  list ind_in_1 = reference_indices(ref_in_1);
	  size_t nb_ind_in_1 = gen_length(ind_in_1);
	  /******/

	  /* pv_in second cell characteristics */     
	  cell cell_in_2 = cell_relation_second_cell(pv_in);
	  reference ref_in_2 = cell_reference(cell_in_2);
	  entity e_in_2 = reference_variable(ref_in_2);
	  list ind_in_2 = reference_indices(ref_in_2);
	  size_t nb_ind_in_2 = gen_length(ind_in_2);
	  /******/

	  pips_debug_pv(3, "considering pv_in:", pv_in);

	  if (same_entity_p(e_kill, e_in_2) && nb_ind_kill <= nb_ind_in_2)
	    {
	      if (cell_relation_second_address_of_p(pv_in) && nb_ind_kill == nb_ind_in_2)
		{
		  /* pointer value relation is still valid */
		  pips_debug(3, "address_of case, and nb_ind_in == nb_ind_kill_2\n");
		  to_be_translated = false;
		}  
	      else
		{
		  pips_debug(3, "second cell is candidate for translation \n");
		  first_p = false;
		  bool inclusion_test_exact_p = false;
		  if ( (nb_ind_kill == nb_ind_in_2 && 
			cell_inclusion_p(cell_in_2, cell_kill, &inclusion_test_exact_p))
		       ||
		       (nb_ind_kill < nb_ind_in_2 && 
			simple_cell_reference_preceding_p(ref_kill, descriptor_undefined, 
							  ref_in_2, descriptor_undefined,
							  transformer_undefined,
							  &exact_preceding_test)))
		    to_be_translated = true;
		  else to_be_translated = false;
		}
	    }
	  else if (same_entity_p(e_kill, e_in_1) && nb_ind_kill <= nb_ind_in_1)
	    {
	      pips_debug(3, "first cell is candidate for translation \n");
	      first_p = true;
	      bool inclusion_test_exact_p = false;
	      if ( (nb_ind_kill == nb_ind_in_1 && 
		    cell_inclusion_p(cell_in_1, cell_kill, &inclusion_test_exact_p) )
		   ||
		   (nb_ind_kill < nb_ind_in_1 && 
		    simple_cell_reference_preceding_p(ref_kill, descriptor_undefined, 
						      ref_in_1, descriptor_undefined,
						      transformer_undefined,
						      &exact_preceding_test)))
		to_be_translated = true;
	      else to_be_translated = false;	      
	    }
	  else
	    {
	      to_be_translated = false;
	    }
      
	  if (to_be_translated)
	    {
	      pips_debug(3, "%s cell must be translated \n", first_p ? "first" : "second");
	      
	      /* This should be made generic */
	  
	      /* we must translate ref_in using the old_values */
	      /* if there is an exact candidate, it is ranked first 
		 and we can use it */
	      if (exact_old_pv != cell_relation_undefined)
		{
		  cell_relation new_pv = simple_pv_translate(pv_in, first_p, exact_old_pv, ctxt);
		  pips_debug_pv(3, "translated to: \n", new_pv);
		  l_out = CONS(CELL_RELATION, new_pv, l_out);		
		}
	  
	      else /* generate a new pv for each element of old_values */
		{
		  FOREACH(CELL_RELATION, old_pv, l_old_values)
		    {
		      cell_relation new_pv = simple_pv_translate(pv_in, first_p, old_pv, ctxt);		  
		      pips_debug_pv(3, "translated to: \n", new_pv);
		      l_out = CONS(CELL_RELATION, new_pv, l_out);
		  
		    } /*  FOREACH(CELL_RELATION, old_pv, l_old_values) */
		} /* else branch of if (exact_first_pv) */ 
	    } /*  if (to_be_translated) */
	  else
	    {
	      pips_debug(3, "non matching case, keep as is\n");
	      l_keep = CONS(CELL_RELATION, copy_cell_relation(pv_in), l_keep);
	    }
	} /* FOREACH (CELL_RELATION, pv_in, l_in) */
   
      list l_tmp = pvs_must_union(l_out, l_keep);
      gen_full_free_list(l_out);
      gen_full_free_list(l_keep);
      l_out = l_tmp;
      
      /* Second, take into account the impact of eff_kill on l_old_values relations. 
	 We only have to keep those which are not completely killed by kill_eff, and 
	 set their approximation to may.
      */
  
      /* first take care of exact_old_pv */
      if (!cell_relation_undefined_p(exact_old_pv))
	{
	  pips_debug(3, "handling exact_old_pv \n");
	  reference ref_old = 
	    cell_reference(cell_relation_second_cell(exact_old_pv));
	  if(same_entity_p(reference_variable(ref_old),e_kill))
	    {
	      pips_debug(3, "exact_old_pv is inverted -> translate\n");
	      cell_relation tmp_old_pv = make_value_of_pointer_value
		(copy_cell(cell_relation_second_cell(exact_old_pv)), 
		 copy_cell(cell_relation_first_cell(exact_old_pv)),
		 cell_relation_approximation_tag(exact_old_pv),
		 make_descriptor_none()); 
	      FOREACH(CELL_RELATION, old_pv, l_old_values)
		{
		  cell_relation new_pv = simple_pv_translate(tmp_old_pv, false, old_pv, ctxt);
		  pips_debug_pv(3, "translating to: \n", new_pv);
		  l_out = CONS(CELL_RELATION, new_pv, l_out);		  
		} 
	      free_cell_relation(tmp_old_pv);
	    }
	  else
	    {
	      pips_debug(3, "exact_old_pv is not inverted -> not need to translate\n");      
	    }
	}
  
      /* Then the other old_values */
      pips_debug(3, "dealing with old values\n");
      FOREACH(CELL_RELATION, pv_old, l_old_values)
	{
	  pips_debug_pv(3, "dealing with pv_old: \n", pv_old);
	  /* we already know that there may be a non-empty intersection with cell_kill */
	  if (app_kill == is_approximation_may)
	    {
	      pips_debug(3, "may kill, just change the approximation\n");
	      cell_relation pv_out = copy_cell_relation(pv_old);
	      cell_relation_approximation_tag(pv_out) = is_approximation_may;
	      l_out = CONS(CELL_RELATION, pv_out, l_out);
	    }
	  else /* some more work is necessary */
	    {
	      cell first_cell_old = cell_relation_first_cell(pv_old);
	      bool exact_inclusion_p = false;
	      bool inclusion_p = cell_inclusion_p(first_cell_old, cell_kill, &exact_inclusion_p);

	      if (inclusion_p && exact_inclusion_p)
		pips_debug(3, "first_cell_old exactly included in cell_kill -> pv_old is killed\n");
	      else
		{
		  /* some more precise work could be done here by computing the difference
		     between the pv_old first cell and cell_kill. I don't know if it would 
		     be really useful.
		     So let us avoid too complex things for the moment.
		  */
		  cell_relation pv_out = copy_cell_relation(pv_old);
		  cell_relation_approximation_tag(pv_out) = is_approximation_may;
		  l_out = CONS(CELL_RELATION, pv_out, l_out);
		}
	    } 
	}
      gen_free_list(l_old_values);
  
    }
  pips_debug_pvs(1, "returning: \n", l_out);
  
  return l_out;
}


cell_relation simple_pv_translate(cell_relation pv_in, bool in_first_p, cell_relation pv_old, pv_context * ctxt)
{
  cell_relation pv_new;

  /* pv_in first or second cell characteristics */     
  cell cell_in = in_first_p ? cell_relation_first_cell(pv_in) : cell_relation_second_cell(pv_in);
  reference ref_in = cell_reference(cell_in);
  entity e_in = reference_variable(ref_in);
  list ind_in = reference_indices(ref_in);
  /******/
  
  /* pv_old characteristics */
  reference ref_old_1 = 
    cell_reference(cell_relation_first_cell(pv_old));
  list ind_old_1 = reference_indices(ref_old_1);
  size_t nb_ind_old_1 = gen_length(ind_old_1);

  reference ref_old_2 = 
    cell_reference(cell_relation_second_cell(pv_old));
  list ind_old_2 = reference_indices(ref_old_2);
  size_t nb_ind_old_2 = gen_length(ind_old_2);
  bool anywhere_old_p = cell_relation_second_address_of_p(pv_old) 
    && entity_all_locations_p(reference_variable(ref_old_2)) ;
  /******/
  
  bool old_first_p = same_entity_p(reference_variable(ref_old_1), e_in); /* is the first cell of pv_old the prefix of ref_in? */

  reference prefix_ref = old_first_p ? ref_old_1 : ref_old_2;
  reference target_ref = old_first_p ? ref_old_2 : ref_old_1;

  reference ref;
  descriptor d;
  bool exact_translation_p;
  int nb_common_indices;

  if (old_first_p && anywhere_old_p)
    {
      cell c1 =  in_first_p ? copy_cell(cell_relation_second_cell(pv_in)) :
	copy_cell(cell_relation_first_cell(pv_in));
      cell c2 = copy_cell(cell_relation_second_cell(pv_old));
      pv_new = make_value_of_pointer_value(c1, c2, 
					     is_approximation_may, make_descriptor_none());	
    }
  else
    {
      if ( (!old_first_p) && cell_relation_second_address_of_p(pv_old))
	{
	  /* act as if there were a [0] indice at the end of ref_old_1 */
	  nb_common_indices = (int) nb_ind_old_1 + 1;
	  
	  (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
	    (ref_in, descriptor_undefined, /* not generic here */
	     target_ref, descriptor_undefined, /* not generic here */
	     nb_common_indices,
	     &ref, &d, &exact_translation_p);
	  
	}
      else
	{
	  nb_common_indices = old_first_p ? (int) nb_ind_old_1 : (int) nb_ind_old_2;
	  
	  if (cell_relation_second_address_of_p(pv_old))
	    (*ctxt->cell_reference_with_address_of_cell_reference_translation_func)
	      (ref_in, descriptor_undefined, /* not generic here */
	       target_ref, descriptor_undefined, /* not generic here */
	       nb_common_indices,
	       &ref, &d, &exact_translation_p);
	  else
	    (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
	      (ref_in, descriptor_undefined, /* not generic here */
	       target_ref, descriptor_undefined, /* not generic here */
	       nb_common_indices,
	       &ref, &d, &exact_translation_p);
	}
      
      tag new_t = (cell_relation_may_p(pv_in) || cell_relation_may_p(pv_old) || !exact_translation_p) 
	? is_approximation_may : is_approximation_exact;
      
      if (in_first_p)
	{
	  if(cell_relation_second_value_of_p(pv_in))
	    pv_new = make_value_of_pointer_value(make_cell_reference(ref),
						 copy_cell(cell_relation_second_cell(pv_in)), 
						 new_t, make_descriptor_none());
	  else 
	    pv_new = make_address_of_pointer_value(make_cell_reference(ref), 
						   copy_cell(cell_relation_second_cell(pv_in)), 
						   new_t, make_descriptor_none());
	}
      else
	{
	  if(cell_relation_second_value_of_p(pv_in))
	    pv_new = make_value_of_pointer_value(copy_cell(cell_relation_first_cell(pv_in)), 
						 make_cell_reference(ref), 
						 new_t, make_descriptor_none());
	  else 
	    pv_new = make_address_of_pointer_value(copy_cell(cell_relation_first_cell(pv_in)), 
						   make_cell_reference(ref), 
						   new_t, make_descriptor_none());
	}
    }
  return pv_new;
}

bool cell_inclusion_p(cell c1, cell c2, bool * exact_inclusion_test_p)
{
  bool res = true;
  *exact_inclusion_test_p = true;

  pips_debug(8, "begin\n");
  if (cell_gap_p(c1) || cell_gap_p(c2))
    pips_internal_error("gap case not handled yet \n");
  
  reference r1 = cell_reference_p(c1) 
    ? cell_reference(c1) : preference_reference(cell_preference(c1));
  entity e1 = reference_variable(r1);
  reference r2 = cell_reference_p(c2) 
    ? cell_reference(c2) : preference_reference(cell_preference(c2));
  entity e2 = reference_variable(r2);

  /* only handle all_locations cells for the moment */
  if (entity_all_locations_p(e1))
    {
      if (entity_all_locations_p(e2))
	{
	  *exact_inclusion_test_p = true;
	  res = true;
	}
      else
	{
	  *exact_inclusion_test_p = true;
	  res = false;
	}
    }
  else
    {
      if (entity_all_locations_p(e2)) /* we cannot have entity_all_locations_p(e1) here */
	{
	  *exact_inclusion_test_p = true;
	  res = true;
	}
      else if (same_entity_p(e1, e2))
	{
	  list inds1 = reference_indices(r1);
	  list inds2 = reference_indices(r2);
	  
	  pips_debug(8, "same entities: %s \n", entity_name(e1));
	  
	  if (gen_length(inds1) == gen_length(inds2))
	    {
	      for(;!ENDP(inds1) && res == true; POP(inds1), POP(inds2))
		{
		  expression exp1 = EXPRESSION(CAR(inds1));
		  expression exp2 = EXPRESSION(CAR(inds2));
		  
		  if (unbounded_expression_p(exp1))
		    {
		      if (!unbounded_expression_p(exp2))
			{
			  res = false;
			  *exact_inclusion_test_p = true;
			}
		    }
		  else if (!unbounded_expression_p(exp2) && !expression_equal_p(exp1, exp2) )
		    {
		      res = false;
		      *exact_inclusion_test_p = true;
		    }
		}
	    }
	  else
	    {
	      *exact_inclusion_test_p = true;
	      res = false;
	    }
	}
      else
	{
	  *exact_inclusion_test_p = true;
	  res = false;
	}
	
    }
  pips_debug(8, "end\n");

  return res;
}

bool cell_intersection_p(cell c1, cell c2, bool * intersection_test_exact_p)
{

  bool res = true;
  *intersection_test_exact_p = true;

  if (cell_gap_p(c1) || cell_gap_p(c2))
    pips_internal_error("gap case not handled yet \n");
  
  reference r1 = cell_reference_p(c1) ? cell_reference(c1) : 
    preference_reference(cell_preference(c1));
  entity e1 = reference_variable(r1);
  reference r2 = cell_reference_p(c2) ? cell_reference(c2) : 
    preference_reference(cell_preference(c2));
  entity e2 = reference_variable(r2);

  /* only handle all_locations cells for the moment */
  if (entity_all_locations_p(e1))
    {
      *intersection_test_exact_p = true;
      res = true;
    }
  else if (entity_all_locations_p(e2)) 
	{
	  *intersection_test_exact_p = true;
	  res = true;
	}
  else if (same_entity_p(e1, e2))
    {
      list inds1 = reference_indices(r1);
      list inds2 = reference_indices(r2);
	  
      if (gen_length(inds1) == gen_length(inds2))
	{
	  for(;!ENDP(inds1) && res == true; POP(inds1), POP(inds2))
	    {
	      expression exp1 = EXPRESSION(CAR(inds1));
	      expression exp2 = EXPRESSION(CAR(inds2));
		  
	      if (!unbounded_expression_p(exp1) 
		  && !unbounded_expression_p(exp2) && 
		  !expression_equal_p(exp1, exp2) )
		{
		  res = false;
		  *intersection_test_exact_p = true;
		}
	    }
	}
      else
	{
	  *intersection_test_exact_p = true;
	  res = false;
	}
    }
  else
    {
      *intersection_test_exact_p = true;
      res = false;
    }
	
  return res;
}


/*
  @input eff is an input effect describing a memory path
  @return a list of effects corresponding to effects on eff cell prefix pointer paths
*/
list simple_effect_intermediary_pointer_paths_effect(effect eff)
{
  pips_debug_effect(5, "input effect : \n", eff);
  list l_res = NIL;
  reference ref = effect_any_reference(eff);
  entity e = reference_variable(ref);
  list ref_inds = reference_indices(ref);
  reference tmp_ref = make_reference(e, NIL);
  type t = basic_concrete_type(entity_type(e));
  bool finished = false;
  
  if (entity_all_locations_p(e))
    return CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);

  while (!finished && !ENDP(ref_inds))
    {
      switch (type_tag(t))
	{
	  
	case is_type_variable:
	  {
	    pips_debug(5," variable case\n");
	    basic b = variable_basic(type_variable(t));
	    size_t nb_dim = gen_length(variable_dimensions(type_variable(t)));
	    
	    /* add to tmp_ref as many indices from ref as nb_dim */
	    for(size_t i = 0; i< nb_dim; i++, POP(ref_inds))
	      {
		reference_indices(tmp_ref) = 
		  gen_nconc(reference_indices(tmp_ref), 
			    CONS(EXPRESSION, 
				 copy_expression(EXPRESSION(CAR(ref_inds))), 
				 NIL));
	      }
	  
	    if (basic_pointer_p(b))
	      {
		pips_debug(5," pointer basic \n");
		if (!ENDP(ref_inds))
		  {
		    pips_debug(5,"and ref_inds is not empty \n");
		    effect tmp_eff = 
		      make_effect(make_cell_reference(copy_reference(tmp_ref)), 
				  copy_action(effect_action(eff)),
				  copy_approximation(effect_approximation(eff)), 
				  make_descriptor_none());
		    l_res = CONS(EFFECT, tmp_eff, l_res);
		    reference_indices(tmp_ref) = 
		      gen_nconc(reference_indices(tmp_ref), 
				CONS(EXPRESSION, 
				     copy_expression(EXPRESSION(CAR(ref_inds))), 
				     NIL));
		    POP(ref_inds);
		  
		    type new_t = copy_type(basic_pointer(b));		  
		    free_type(t);
		    t = new_t;
		  }
		else 
		  finished = true;
	      }
	    else if (basic_derived_p(b))
	      {
		pips_debug(5,"derived basic \n");
		type new_t = basic_concrete_type(entity_type(basic_derived(b)));
		free_type(t);
		t = new_t;
	      }
	    else
	      finished = true;
	  }
	  break;
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  {
	    pips_debug(5,"struct union or enum type \n");
		
	    /* add next index */
	    expression field_exp = EXPRESSION(CAR(ref_inds));
	    reference_indices(tmp_ref) = 
	      gen_nconc(reference_indices(tmp_ref), 
			CONS(EXPRESSION, 
			     copy_expression(field_exp), 
			     NIL));
	    POP(ref_inds);
	    entity field_ent = expression_to_entity(field_exp);
	    pips_assert("expression is a field entity\n", !entity_undefined_p(field_ent));
	    type new_t = basic_concrete_type(entity_type(field_ent));
	    free_type(t);
	    t = new_t;
	  }
	  break;
	default:
	    pips_internal_error("unexpected type tag\n");
      
	}
    }
  free_type(t);
  
  return l_res;
}

/**
   @brief find pointer_values in l_in which give (possible or exact) paths 
          equivalent to eff. 
   @param eff is the considered input path.
   @param l_in is the input pointer values list.
   @param exact_aliased_pv gives an exact equivalent path found in l_in if it exists.
   @param l_in_remnants contains the elemnts of l_in which are neither 
          exact_aliased_pv nor in the returned list.

   @return a list of elements of l_in which give (possible or exact) paths 
           equivalent to eff, excluding exact_aliased_pv if one exact equivalent 
           path can be found in l_in.  
 */
list effect_find_equivalent_pointer_values(effect eff, list l_in, 
					   cell_relation * exact_aliased_pv, 
					   list * l_in_remnants)
{

  pips_debug_pvs(1,"begin, l_in = \n", l_in);
  pips_debug_effect(1, "and eff:\n", eff);

  /* eff characteristics */
  cell eff_cell = effect_cell(eff);
  reference ref = effect_any_reference(eff);
  list ind = reference_indices(ref);
  /******/

 /* first, search for the (exact/possible) values of eff cell in l_in */
  /* we search for the cell_relations where ref appears
     as a first cell, or the exact value_of pointer_values where ref appears as 
     a second cell. If an exact value_of relation is found, it is retained in 
     exact_aliased_pv  
  */
  *l_in_remnants = NIL;
  *exact_aliased_pv = cell_relation_undefined;
  list l_res = NIL;

  FOREACH(CELL_RELATION, pv_in, l_in)
    {
      cell first_cell_in = cell_relation_first_cell(pv_in);
      cell second_cell_in = cell_relation_second_cell(pv_in);
      bool intersection_test_exact_p = false;
      bool inclusion_test_exact_p = true;
      
      pips_debug_pv(4, "considering: \n", pv_in);
      if (cell_intersection_p(eff_cell, first_cell_in, &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with first cell (%sexact)\n", 
		     intersection_test_exact_p? "": "non ");
	  if (cell_relation_exact_p(pv_in) 
	      && intersection_test_exact_p
	      && cell_inclusion_p(first_cell_in, eff_cell, &inclusion_test_exact_p)
	      && inclusion_test_exact_p)
	    {
	      pips_debug(4, "exact value candidate found\n");
	      *exact_aliased_pv = pv_in;
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else if(cell_relation_second_value_of_p(pv_in) 
	      && cell_intersection_p(eff_cell, second_cell_in, &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with second value_of cell (%sexact)\n", 
		     intersection_test_exact_p? "": "non ");
	  if(cell_relation_exact_p(pv_in) 
	      && intersection_test_exact_p
	      && cell_inclusion_p(second_cell_in, eff_cell, &inclusion_test_exact_p)
	      && inclusion_test_exact_p)
	    {
	      pips_debug(4, "exact value candidate found\n");
	      *exact_aliased_pv = pv_in;
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else
	{
	  pips_debug(4, "remnant\n");
	  *l_in_remnants = CONS(CELL_RELATION, pv_in, *l_in_remnants);	
	}
    }
  pips_debug_pvs(3, "l_in_remnants: \n", *l_in_remnants);
  pips_debug_pvs(3, "l_res: \n", l_res);
  pips_debug_pv(3, "*exact_aliased_pv: \n", *exact_aliased_pv);

  return l_res;
  
}

/**
   @brief find all paths equivalent to eff cell in l_pv by performing a transitive closure
   @param eff is the input effect
   @param l_pv is the list of current pointer_values relations
   @param ctxt is the pv analysis context
   @return a list of effects whose cells are equivalent to eff_kill cell according to l_pv.
           Their approximation does not depend on the approximation of the input effect,
	   but only on the exactness of the finding process.

 */
list effect_find_aliased_paths_with_pointer_values(effect eff, list l_pv, pv_context *ctxt)
{
  list l_res = NIL;
  list l_remnants = l_pv;
  reference eff_ref = effect_any_reference(eff);
  bool anywhere_p = false;

  pips_debug_effect(5, "begin with eff : \n", eff);
  pips_debug_pvs(5, "and l_pv : \n", l_pv);

  if (anywhere_effect_p(eff)) /* should be turned into entity_abstract_location_p */
    {
      pips_debug(5, "anywhere case\n");
      
      return (NIL);
    }
  else
    {

      /* first we must find in eff_kill intermediary paths to pointers */
      /* not generic here */
      list l_intermediary = simple_effect_intermediary_pointer_paths_effect(eff);
      pips_debug_effects(5, "intermediary paths to eff: \n", l_intermediary);
      
      /* and find if this gives equivalent paths in l_pv */
      FOREACH(EFFECT, eff_intermediary, l_intermediary)
	{
	  pips_debug_effect(5, "considering intermediary path : \n", eff_intermediary);
	  list tmp_l_remnants = NIL;
	  cell_relation pv_exact = cell_relation_undefined;
	  list l_equiv = effect_find_equivalent_pointer_values(eff_intermediary, l_remnants, 
							       &pv_exact, 
							       &tmp_l_remnants);
	  if (!cell_relation_undefined_p(pv_exact))
	    l_equiv = CONS(CELL_RELATION, pv_exact, l_equiv);
	  l_remnants = tmp_l_remnants;
	  pips_debug_pvs(5, "list of equivalent pvs \n", l_equiv);
	  
	  reference ref_intermediary = effect_any_reference(eff_intermediary);
	  entity ent_intermediary = reference_variable(ref_intermediary);
	  descriptor d_intermediary = effect_descriptor(eff_intermediary);
	  int nb_common_indices = (int) gen_length(reference_indices(ref_intermediary));
	  
	  FOREACH(CELL_RELATION, pv_equiv, l_equiv)
	    {
	      reference ref;
	      descriptor d;
	      bool exact_translation_p;
	      
	      pips_debug_pv(5, "translating eff using pv: \n", pv_equiv);
	      
	      /* this is valid only if the first value_of corresponds to eff_intermediary */
	      reference pv_equiv_first_ref = cell_reference(cell_relation_first_cell(pv_equiv));
	      reference pv_equiv_second_ref = cell_reference(cell_relation_second_cell(pv_equiv));
	      
	      if (same_entity_p(ent_intermediary, reference_variable(pv_equiv_first_ref))
		  && (gen_length(reference_indices(ref_intermediary)) 
		      == gen_length(reference_indices(pv_equiv_first_ref))))
		{
		  /* use second cell as equivalent value for intermediary path */
		  if (cell_relation_second_value_of_p(pv_equiv))
		    {
		      (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
			(eff_ref, descriptor_undefined, /* not generic here */
			 cell_reference(cell_relation_second_cell(pv_equiv)),
			 descriptor_undefined, /* not generic here */
			 nb_common_indices,
			 &ref, &d, &exact_translation_p);
		    }
		  else /* cell_relation_second_address_of_p is true */
		    {
		      (*ctxt->cell_reference_with_address_of_cell_reference_translation_func)
			(eff_ref, descriptor_undefined, /* not generic here */
			 cell_reference(cell_relation_second_cell(pv_equiv)),
			 descriptor_undefined, /* not generic here */
			 nb_common_indices,
			 &ref, &d, &exact_translation_p);
		    }
		}
	      else /* use first cell as equivalent value for intermediary path  */
		{
		  pips_assert("pv_equiv must be value_of here\n", 
			      cell_relation_second_value_of_p(pv_equiv));
		  
		  (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
		    (eff_ref, descriptor_undefined, /* not generic here */
		     cell_reference(cell_relation_first_cell(pv_equiv)),
		     descriptor_undefined, /* not generic here */
		     nb_common_indices,
		     &ref, &d, &exact_translation_p);
		}
	      exact_translation_p = exact_translation_p && cell_relation_exact_p(pv_equiv);
	      
	      effect eff_alias = make_effect(make_cell_reference(ref),
					     copy_action(effect_action(eff_intermediary)),
					     exact_translation_p ? 
					     make_approximation_must()
					     : make_approximation_may(), make_descriptor_none());
	      pips_debug_effect(5, "resulting effect \n", eff_alias);
	      if (anywhere_effect_p(eff_alias))
		{
		  gen_full_free_list(l_res);
		  l_res = CONS(EFFECT, eff_alias, NIL);
		  anywhere_p = true;
		}
	      else
		{
		  l_res = CONS(EFFECT, eff_alias, l_res);	  
		  
		}
	    }
	}
      
      if (!anywhere_p)
	{
	  pips_debug_effects(5, "l_res after first phase : \n", l_res);
      
	  /* Then we must find  if there are address_of second cells 
	     which are preceding paths of eff path 
	     in which case they must be used to generate other aliased paths 
	  */
	  list l_remnants_2 = NIL;
	  FOREACH(CELL_RELATION, pv_remnant, l_remnants)
	    {
	      reference pv_remnant_second_ref = 
		cell_reference(cell_relation_second_cell(pv_remnant));
	      bool exact_preceding_test = true;
	  
	      pips_debug_pv(5, "considering pv: \n", pv_remnant);
	  
	      if (cell_relation_second_address_of_p(pv_remnant) 
		  && same_entity_p(reference_variable(eff_ref), 
				   reference_variable(pv_remnant_second_ref))
		  && (gen_length(reference_indices(eff_ref)) 
		      >= gen_length(reference_indices(pv_remnant_second_ref)))
		  && simple_cell_reference_preceding_p(pv_remnant_second_ref, descriptor_undefined, 
						       eff_ref, descriptor_undefined,
						       transformer_undefined,
						       &exact_preceding_test))
		{
		  reference ref;
		  descriptor d;
		  bool exact_translation_p;
	      
		  pips_debug(5, "good candidate (%sexact)\n",exact_preceding_test? "":"non "); 
		  /* for the translation, add a dereferencing_dimension to pv_remnant_first_cell */
		  reference new_ref = copy_reference
		    (cell_reference(cell_relation_first_cell(pv_remnant)));
		  int nb_common_indices = (int) gen_length(reference_indices(pv_remnant_second_ref));
		  /* not generic here */
		  reference_indices(new_ref) = gen_nconc(reference_indices(new_ref),
							 CONS(EXPRESSION,
							      int_to_expression(0),
							      NIL));
	      
		  (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
		    (eff_ref, descriptor_undefined, /* not generic here */
		     new_ref,
		     descriptor_undefined, /* not generic here */
		     nb_common_indices,
		     &ref, &d, &exact_translation_p);
	      
		  exact_translation_p = exact_translation_p && cell_relation_exact_p(pv_remnant);
	      
		  effect eff_alias = make_effect(make_cell_reference(ref),
						 make_action_write_memory(),
						 exact_translation_p && exact_preceding_test ? 
						 make_approximation_must()
						 : make_approximation_may(), make_descriptor_none());
		  free_reference(new_ref);
		  pips_debug_effect(5, "resulting effect \n", eff_alias);
		  l_res = CONS(EFFECT, eff_alias, l_res);	  
		  
		}
	      else 
		{
		  l_remnants_2 = CONS(CELL_RELATION, pv_remnant, l_remnants_2);
		}
	  
	    } /* FOREACH */


	  l_remnants = l_remnants_2;
	} /* if (!anywhere_p)*/
      if (!ENDP(l_remnants))
	{
	  pips_debug(5, "recursing to find aliases to aliased effect...\n");
	  pips_debug_effects(5, "l_res before recursing : \n", l_res);
	  list l_recurs = NIL;
	  FOREACH(EFFECT, eff_alias, l_res)
	    {
	      l_recurs = gen_nconc(l_recurs, 
				   effect_find_aliased_paths_with_pointer_values(eff_alias, 
										 l_remnants, 
										 ctxt));
	    }
	  l_res = gen_nconc(l_recurs, l_res);
	} 
      

    } /* else branche of if (anywhere_effect_p(eff))*/


  pips_debug_effects(5, "returning : \n", l_res);
  return l_res;

}


/* 
   @brief change the reference indices into store independent expressions
   
   @param r is an input reference, and may be modified by side effect.
   @return true if an element of the reference has been changed into an unbounded expression.
*/
reference simple_reference_to_store_independent_reference(reference r, bool * changed_p)
{
  *changed_p = false;
  
  list l_inds = reference_indices(r);
  list cind = list_undefined;

  for(cind = l_inds; !ENDP(cind); POP(cind)) 
    {
      expression se = EXPRESSION(CAR(cind));
      
      if(!extended_integer_constant_expression_p(se)) 
	{
	  if(!unbounded_expression_p(se)) 
	    {	      
	      /* it may still be a field entity */
	      if (!(expression_reference_p(se) &&
		    entity_field_p(expression_variable(se))))
		{
		  *changed_p = true;
		  free_expression(se);
		  EXPRESSION_(CAR(cind)) = make_unbounded_expression();
		}
	    }
	}
    }
  
  return r;
}


/* 
   @brief change the cell indices into store independent expressions
   
   @param c is an input cell, and may be modified by side effect.
   @return true if an element of the cell has been changed into an unbounded expression.
*/
cell simple_cell_to_store_independent_cell(cell c, bool * changed_p)
{
  pips_assert("gaps not handled yet\n", !cell_gap_p(c));
  reference r = cell_reference_p(c) ? cell_reference(c) : preference_reference(cell_preference(c));

  cell_reference(c) = simple_reference_to_store_independent_reference(r, changed_p);
  return c;
  
}

/* 
   @brief change each element of simple pointer values input list into a store independent pointer value.
   
   @param l_pv is the input list of simple pointer values
   @param t is unused, but is here for homogeneity purposes
*/
cell_relation simple_pv_composition_with_transformer(cell_relation pv, transformer  __attribute__ ((unused)) t)
{
  cell c1 = cell_relation_first_cell(pv);
  cell c2 = cell_relation_second_cell(pv);

  bool b1, b2;

  cell_relation_first_cell(pv) = simple_cell_to_store_independent_cell(c1, &b1);
  cell_relation_second_cell(pv) = simple_cell_to_store_independent_cell(c2, &b2);
  
  if (cell_relation_exact_p(pv) && (b1 || b2))
    cell_relation_approximation_tag(pv) = is_approximation_may;
  return pv;
}

/* 
   @brief report the impact of store modification modelized by the input transformer onto the input list of pointer values
   
   @param l_pv is the input list of pointer values
   @param t is the transfomer that modelizes the store modification
   @param ctxt is a pointer on the pointer value analysis context holder.
*/
list pvs_composition_with_transformer(list l_pv, transformer t, pv_context * ctxt)
{
  FOREACH(CELL_RELATION, pv, l_pv)
    {
     pv = (*ctxt->pv_composition_with_transformer_func)(pv, t);
    }

  return l_pv;
}

/*
  @brief tests the syntactic equality of two pointer_value relations
  @param pv1 is a pointer_value
  @param pv2 is another pointer value
  @return true if the input pointer values are syntactically equal.

  if both pvs are value_of pvs, they are considered equal if their first cells are equal
  and second cells are equal but also if the first cell of the first pv is equal to
  the second cell of the second pv and conversely.
 */
bool pv_syntactically_equal_p(cell_relation pv1, cell_relation pv2)
{

  if (cell_relation_approximation_tag(pv1) != cell_relation_approximation_tag(pv2))
    return false;

  bool value_of_1_p = cell_relation_second_value_of_p(pv1);
  bool value_of_2_p = cell_relation_second_value_of_p(pv1);

  if ( (value_of_1_p && !value_of_2_p) || (value_of_2_p && !value_of_1_p))
    return false;

  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  int n_first_first = cell_compare(&c_first_1, &c_first_2);

  if (n_first_first == 0)
    {
      int n_second_second = cell_compare(&c_second_1, &c_second_2);
      
      if (n_second_second != 0)
	return false;
    }
  else
    {
      if (!value_of_1_p)
	return false;
      else /* value_of pvs, try to see if their cells are inverted */
	{
	  int n_first_second = cell_compare(&c_first_1, &c_second_2);
	  if (n_first_second == 0)
	    {
	      int n_second_first = cell_compare(&c_second_1, &c_first_2);
	      
	      if (n_second_first != 0)
		return false;
	    }	
	  else
	    return false;
	  
	}
    }
  
  descriptor d1 = cell_relation_descriptor(pv1);
  descriptor d2 = cell_relation_descriptor(pv1);

  if (descriptor_none_p(d1) && descriptor_none_p(d2))
    {
      return true;
    }
  else
    pips_internal_error("Convex pointer_values not implemented yet\n");
  
  return false;
}

/*
  @brief computes the union of two pointer_values list
  @param l_pv1 is the first list of pointer_values
  @param l_pv2 is the second list of pointer_values
  @return a new list of pointer values

  
 */
list pvs_must_union(list l_pv1, list l_pv2)
{
  list l_res = NIL;
  FOREACH(CELL_RELATION, pv2, l_pv2)
    {
      list l1 = l_pv1;
      bool found = false;
      while(!found && !ENDP(l1))
	{
	  cell_relation pv1 = CELL_RELATION(CAR(l1));
	  if (pv_syntactically_equal_p(pv1, pv2))
	    found = true;
	  POP(l1);	    
	}
      if (!found)
	l_res = CONS(CELL_RELATION, copy_cell_relation(pv2), l_res);
    }
  
  FOREACH(CELL_RELATION, pv1, l_pv1)
    {
      l_res = CONS(CELL_RELATION, copy_cell_relation(pv1), l_res);
    }
  return l_res;
}
