/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: proper_effects_engine.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of
 * all types of proper effects and proper references.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "pipsdbm.h"
#include "resources.h"

#include "pointer_values.h"
#include "effects-generic.h"

/* For debuging

static void debug_ctxt(string s, transformer t)
{
  Psysteme p;
  fprintf(stderr, "context %p at %s\n", t, s);
  if (transformer_undefined_p(t))
    fprintf(stderr, "UNDEFINED...");
  else
    {
      p = predicate_system(transformer_relation(t));
      fprintf(stderr, "%p: %d/%d\n", p, sc_nbre_egalites(p), sc_nbre_inegalites(p));
      sc_syst_debug(p);
      assert(sc_weak_consistent_p(p));
    }
}
*/

/************************************************ TO CONTRACT PROPER EFFECTS */

static bool contract_p = true;

void
set_contracted_proper_effects(bool b)
{
    contract_p = b;
}


/**************************************** LOCAL STACK FOR LOOP RANGE EFFECTS */

/* Effects on loop ranges have to be added to inner statements to model
 * control dependances (see loop filter for PUSH).
 */

DEFINE_LOCAL_STACK(current_downward_cumulated_range_effects, effects)

void proper_effects_error_handler()
{
    error_reset_effects_private_current_stmt_stack();
    error_reset_effects_private_current_context_stack();
    error_reset_current_downward_cumulated_range_effects_stack();
}

static list
cumu_range_effects()
{
      list l_cumu_range = NIL;

      if(! current_downward_cumulated_range_effects_empty_p())
      {
	  l_cumu_range = effects_effects
			  (current_downward_cumulated_range_effects_head());
      }
      return(l_cumu_range);
}

static void
free_cumu_range_effects()
{
    if(! current_downward_cumulated_range_effects_empty_p())
	free_effects(current_downward_cumulated_range_effects_head());
}


/************************************************************** EXPRESSSIONS */

/* list generic_proper_effects_of_range(range r, context)
 * input    : a loop range (bounds and stride) and the context.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :
 */
list generic_proper_effects_of_range(range r)
{
    list le;
    expression el = range_lower(r);
    expression eu = range_upper(r);
    expression ei = range_increment(r);

    pips_debug(5, "begin\n");

    le = generic_proper_effects_of_expression(ei);
    le = gen_nconc(generic_proper_effects_of_expression(eu), le);
    le = gen_nconc(generic_proper_effects_of_expression(el), le);

    pips_debug(5, "end\n");
    return(le);
}


static list generic_r_proper_effects_of_derived_reference(effect input_eff, type input_type)
{
  basic b = variable_basic(type_variable(input_type));
  list l_dim =  variable_dimensions(type_variable(input_type));
  list le = NIL;

  pips_debug_effect(8, "input effects:\n", input_eff);

  pips_assert("input entity type basic must be derived", basic_derived_p(b));

  // we first add as many unbounded indices as there are dimensions in the type.
  for(int i=0; i< (int) gen_length(l_dim); i++)
    (*effect_add_expression_dimension_func)(input_eff, make_unbounded_expression());

  pips_debug(8, "type of basic derived : %s\n",words_to_string(words_type(entity_type(basic_derived(b)), NIL, false)));
  list l_fields = type_fields(entity_type(basic_derived(b)));

  FOREACH(ENTITY, f, l_fields)
    {
      type current_type = entity_basic_concrete_type(f);
      basic current_basic = variable_basic(type_variable(current_type));
      effect current_eff = (*effect_dup_func)(input_eff);

      // we add the field index
      effect_add_field_dimension(current_eff, f);

      switch (basic_tag(current_basic))
	{
	case is_basic_derived:
	  if (type_enum_p(entity_type(basic_derived(current_basic))))
	    {
	      pips_debug(8, "enum case \n");
	      le = gen_nconc(CONS(EFFECT, current_eff, NIL), le);
	    }
	  else
	    {
	      pips_debug(8, "derived case : recursing\n");
	      le = gen_nconc(generic_r_proper_effects_of_derived_reference(current_eff, current_type),le);
	      effect_free(current_eff);
	    }
	  break;
	case is_basic_typedef:
	  // should not happen
	  pips_internal_error("typedef case should not be possible here! ");
	  break;
	default:
	  {
	    // we have reached a leaf : we just have to add as many unbounded indices as there are dimensions in the field type.
	    list l_dim =  variable_dimensions(type_variable(current_type));
	    pips_debug(8, "default case\n");
	    for(int i=0; i< (int) gen_length(l_dim); i++)
	      (*effect_add_expression_dimension_func)(current_eff, make_unbounded_expression());

	    le = gen_nconc(CONS(EFFECT, current_eff, NIL), le);
	  }
	}
    }
  pips_debug_effects(8, "output effects:\n", le);

  return le;
}

list generic_proper_effects_of_derived_reference(reference ref, bool write_p)
{
  type t = reference_to_type(ref);
  basic b = variable_basic(type_variable(t));
  list le = NIL;

  pips_debug(8, "type of basic derived : %s\n",words_to_string(words_type(entity_type(basic_derived(b)), NIL, false)));

  /* should'nt it be something more direct here ? ?*/
  effect eff = (*reference_to_effect_func)
    (ref, write_p? make_action_write_memory() : make_action_read_memory(), true);

  if (type_enum_p(entity_type(basic_derived(b))))
    le = CONS(EFFECT, eff, NIL);
  else
    {
      le = generic_r_proper_effects_of_derived_reference(eff, t);
      le = gen_nreverse(le);
    }

  return le;
}

list generic_intermediary_proper_effects_of_reference(reference ref)
{
  list le = NIL;
  entity ent = reference_variable(ref);
  type t = entity_basic_concrete_type(ent);
  list l_ind_orig = reference_indices(ref);
  list l_ind = l_ind_orig;

  pips_debug(7, "begin \n");
  /* there should be a while here, I guess. Well work for tomorrow */

  if (type_variable_p(t))
    {
      variable v = type_variable(t);
      basic b = variable_basic(v);

      pips_debug(4, "reference %s to entity %s of basic %s and"
		 " number of dimensions %d.\n",
		 words_to_string(words_reference(ref,NIL)),
		 entity_name(ent),
		 basic_to_string(variable_basic(v)),
		 (int) gen_length(variable_dimensions(v)));

      if (basic_pointer_p(b))
	{
	  reference read_ref = make_reference(ent, NIL);
	  list l_dim_tmp = variable_dimensions(v);

	  pips_debug(8, "it's a pointer type. l_dim_tmp = %d, "
		     "l_inds_tmp = %d\n",
		     (int) gen_length(l_dim_tmp),
		     (int) gen_length(l_ind));

	  /* while there is non partially subscripted references to pointers */
	  while((basic_pointer_p(variable_basic(v)))
		&& gen_length(l_dim_tmp) < gen_length(l_ind) )
	    {
	      effect eff_read;

	      /* first we add the indices corresponding to the current
		 array dimensions if any
	      */

	      pips_debug(8, "l_dim_tmp = %d, l_inds_tmp = %d\n",
			 (int) gen_length(l_dim_tmp),
			 (int) gen_length(l_ind));

	      while (!ENDP(l_dim_tmp))
		{
		  reference_indices(read_ref)=
		    gen_nconc(reference_indices(read_ref),
			      CONS(EXPRESSION, copy_expression(EXPRESSION(CAR(l_ind))),
				   NIL));
		  POP(l_dim_tmp);
		  POP(l_ind);
		}

	      pips_debug(4, "adding a read effect on reference %s\n",
			 words_to_string(words_reference(read_ref,NIL)));

	      eff_read = (*reference_to_effect_func)(copy_reference(read_ref),
						     make_action_read_memory(),
						     false);
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));
	      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));

	      v = type_variable(basic_pointer(variable_basic(v)));
	      l_dim_tmp = variable_dimensions(v);

	      /* if there are remaining indices, there is necessarily an index
		 to reach the pointed dimension */
	      if(!ENDP(l_ind))
		{
		  pips_debug(4, "adding an index for pointer dimension \n");
		  reference_indices(read_ref)=
		    gen_nconc(reference_indices(read_ref),
			      CONS(EXPRESSION,
				   copy_expression(EXPRESSION(CAR(l_ind))),
				   NIL));
		  POP(l_ind);
		}
	    }
	  free_reference(read_ref);
	}

    }  /* if (type_variable_p(t)) */
  gen_nreverse(le);
  le = gen_nconc(generic_proper_effects_of_expressions(l_ind), le);

  pips_debug_effects(7, "end with \n", le);
  return le;
}




/**
 @param ref a reference.
 @param *pme : a pointer on the main effect corresponding to the reference.
               if there is no effect (partially subscripted array for instance)
	       then *pme is set to effect undefined. If the main effect
               could not be computed, *pme is set to a new anywhere effect.
 @param write_p : true if the main effect is write, false otherwise.
 @param allow_read_on_pme : true if we want to allow the generation of
                read effects on reference even if it's a partially subscripted
		array. The default value should be false, but it is useful
                to set it to true when recusrsively building pme).
 @return : a list of read effects corresponding to the intermediate reads
           performed to access the main memory location.

*/
list generic_p_proper_effect_of_reference(reference ref,
					  effect *pme,
					  bool write_p,
					  bool allow_partials_on_pme)
{
  list le = NIL; /* list of read effects */

  entity ent = reference_variable(ref);
  list l_inds = reference_indices(ref);
  list l_inds_tmp = l_inds;

  type t = entity_basic_concrete_type(ent);

  *pme = effect_undefined;

  /* now we generate the read effects on intermediate pointer dimensions */
  /* if the entity reference is a pointer, then we scan the dimensions
     until we reach a non-pointer basic.
  */

  if (type_variable_p(t) && c_module_p(get_current_module_entity()))
    {
      variable v = type_variable(t);
      basic b = variable_basic(v);

      pips_debug(4, "reference %s to entity %s of basic %s and"
		 " number of dimensions %d.\n",
		 words_to_string(words_reference(ref,NIL)),
		 entity_name(ent),
		 basic_to_string(variable_basic(v)),
		 (int) gen_length(variable_dimensions(v)));

      if (basic_pointer_p(b))
	{
	  reference read_ref = make_reference(ent, NIL);
	  list l_dim_tmp = variable_dimensions(v);

	  pips_debug(8, "it's a pointer type. l_dim_tmp = %d, "
		     "l_inds_tmp = %d\n",
		     (int) gen_length(l_dim_tmp),
		     (int) gen_length(l_inds_tmp));

	  /* while there is non partially subscripted references to pointers */
	  while((basic_pointer_p(variable_basic(v)))
		&& gen_length(l_dim_tmp) < gen_length(l_inds_tmp) )
	    {
	      effect eff_read;

	      /* first we add the indices corresponding to the current
		 array dimensions if any
	      */

	      pips_debug(8, "l_dim_tmp = %d, l_inds_tmp = %d\n",
			 (int) gen_length(l_dim_tmp),
			 (int) gen_length(l_inds_tmp));

	      while (!ENDP(l_dim_tmp))
		{
		  reference_indices(read_ref)=
		    gen_nconc(reference_indices(read_ref),
			      CONS(EXPRESSION, copy_expression(EXPRESSION(CAR(l_inds_tmp))),
				   NIL));
		  POP(l_dim_tmp);
		  POP(l_inds_tmp);
		}

	      pips_debug(4, "adding a read effect on reference %s\n",
			 words_to_string(words_reference(read_ref,NIL)));

	      eff_read = (*reference_to_effect_func)(copy_reference(read_ref),
						     make_action_read_memory(),
						     false);
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));
	      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));

	      v = type_variable(basic_pointer(variable_basic(v)));
	      l_dim_tmp = variable_dimensions(v);

	      /* if there are remaining indices, there is necessarily an index
		 to reach the pointed dimension */
	      if(!ENDP(l_inds_tmp))
		{
		  pips_debug(4, "adding an index for pointer dimension \n");
		  reference_indices(read_ref)=
		    gen_nconc(reference_indices(read_ref),
			      CONS(EXPRESSION,
				   copy_expression(EXPRESSION(CAR(l_inds_tmp))),
				   NIL));
		  POP(l_inds_tmp);
		}
	    }
	  free_reference(read_ref);
	}

      /* no read or write effects on partial array if
	 allow_partials_on_pme is false */
      if((
	  basic_pointer_p(b) &&
	  ( gen_length(variable_dimensions(v)) <= gen_length(l_inds)
	    || allow_partials_on_pme
	    )
	  )
	 || (!basic_pointer_p(b) &&
	  (allow_partials_on_pme ||
	   gen_length(variable_dimensions(v)) == gen_length(l_inds))))
	{
        action a = write_p? make_action_write_memory() : make_action_read_memory();
	  *pme = (*reference_to_effect_func)(ref, a, true);
      free_action(a);
	  pips_assert("*pme is wekly consistent", region_weakly_consistent_p(*pme));
	}
    }
  else
    {
      /* Just compute the main memory effect of the reference

	 This should maybe be refined ? */

      /* If the entity referenced is a function, we do not want a
	 memory effect since it is a constant. Note: we end up here
	 because its type as been converted to a pointer to a
	 function above. */
      entity rv= reference_variable(ref);
      type rvt = ultimate_type(entity_type(rv));

      if(type_functional_p(rvt))
	*pme = effect_undefined;
      else
	*pme = (*reference_to_effect_func)
	  (ref, write_p? make_action_write_memory() : make_action_read_memory(), true);

      if(!effect_undefined_p(*pme))
	pips_assert("*pme is weakly consistent", region_weakly_consistent_p(*pme));
    }

  /* we must add the read effects on the indices ; these reads are performed
   before the main effect */
  pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));
  le = gen_nconc(generic_proper_effects_of_expressions
		 (reference_indices(ref)), le);
  pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));


  /* and there is a declaration effect */
  if(!get_bool_property("MEMORY_EFFECTS_ONLY"))
    le = CONS(EFFECT, make_declaration_effect(ent,false), le);
  ifdebug(4)
    {
      if(effect_undefined_p(*pme))
	pips_debug(4, "ending no main effect "
		   "(e.g. a non-subscribed reference to an array)\n");
      else
	pips_debug_effect(4, "ending with main effect : \n", *pme);

      pips_debug_effects(4, "and intermediate read effects : \n",le);
    }

  return le;
}



/* list generic_proper_effects_of_reference(reference ref, bool written_p)
 * input    : a reference and a boolean true if it is a written reference.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is either read or written.
 */
list generic_proper_effects_of_reference(reference ref, bool written_p)
{
  list le = NIL;
  entity v = reference_variable(ref);

  pips_debug(3, "begin with reference %s\n",
	     words_to_string(words_reference(ref,NIL)));
  pips_assert("no effect on entity fields\n",!entity_field_p(v));
  transformer context;

  if ( !effects_private_current_context_stack_initialized_p()
       || effects_private_current_context_empty_p())
    context = transformer_undefined;
  else {
    context = effects_private_current_context_head();
  }

  if (! (*empty_context_test)(context))
    {
      effect eff;
      type ref_type = reference_to_type(ref);
      if (type_variable_p(ref_type))
	{
	  variable ref_type_var = type_variable(ref_type);
	  if (basic_derived_p(variable_basic(ref_type_var)) && ENDP(variable_dimensions(ref_type_var)))
	    {
	      list lint = generic_intermediary_proper_effects_of_reference(ref);
	      le = generic_proper_effects_of_derived_reference(ref, written_p);
	      le = gen_nconc(lint, le);
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));

	        /* Environment and declaration type effects */
	      if(!get_bool_property("MEMORY_EFFECTS_ONLY")) {
		/* May not be generic enough altough contexts seem useless for
		   environment effects */
		/* FI: these effects do not seem to always combine. A statement
		   with a read and a write of v seems to end with two reference
		   effects. See Bootstrap/iand01.f, assuming all effects are
		   computed, of course. */
		effect re = make_declaration_effect(v, false); // reference effect
		//type vt = entity_type(v);
		le = CONS(EFFECT, re, le);

		/*
		  if(typedef_type_p(vt)) {
		  entity te = basic_typedef(variable_basic(type_variable(vt)));
		  effect tre = make_declaration_effect(te, false); // type
		  // reference effect
		  le = CONS(EFFECT, tre, le);
		  }
		*/
	      }
	    }
	  else
	    {
	      /* environment effects are also computed here */
	      le =  generic_p_proper_effect_of_reference(ref, &eff, written_p,
							 false);
	      pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));

	      if (!effect_undefined_p(eff))
		{
		  le = gen_nconc(le, CONS(EFFECT, eff, NIL));
		  pips_assert("le is weakly consistent", regions_weakly_consistent_p(le));
		}
	    }
	  free_type(ref_type);
	}
	else
	  pips_internal_error("case not handled yet ");
      (*effects_precondition_composition_op)(le, context);
    }


  pips_debug(3, "end\n");
  return(le);
}


/* list generic_proper_effects_of_read_reference(reference ref)
 * input    : a reference that is read.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is read
 */
list generic_proper_effects_of_read_reference(reference ref)
{
  list le = NIL;

  le = generic_proper_effects_of_reference(ref, false);

  return(le);
}

/* list proper_effects_of_written_reference(reference ref)
 * input    : a reference that is written.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is written
 */
list generic_proper_effects_of_written_reference(reference ref)
{
  list le = NIL;

  le = generic_proper_effects_of_reference(ref, true);

  return(le);
}


static list generic_proper_effects_of_complex_address_field_op(list l_args, list *l_pme, int write_p)
{
  list le = NIL;
  /* first get an effect on the structure */
  /* is the structure a constant as in {0, 1}.im */
  expression struct_exp = EXPRESSION(CAR(l_args));
  // very inefficient. g_p_e_o_c_a_e should convey a way to say that it's argument is constant
 /*  list le1 = generic_proper_effects_of_expression(struct_exp); */

/*   if (ENDP(le1)) */
/*     { */
/*       pips_debug(8, "field operator applied on a constant expression -> no effect"); */
/*     } */
/*   else */
/*     { */
/*      gen_full_free_list(le1);*/
      le = generic_proper_effects_of_complex_address_expression
	(struct_exp, l_pme, write_p);

      /* and add the field */
      FOREACH(EFFECT, pme, *l_pme)
	{
	  if(!effect_undefined_p(pme) && !anywhere_effect_p(pme))
	    {
	      expression field_exp = EXPRESSION(CAR(CDR(l_args)));
	      syntax s = expression_syntax(field_exp);
	      entity f;
	      ifdebug(1) pips_assert("e2 is a reference", syntax_reference_p(s));
	      f = reference_variable(syntax_reference(s));
	      /* we extend *pme by adding a dimension corresponding
	       * to the field */
	      effect_add_field_dimension(pme,f);
	    }
	}
      /*}*/
  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);

  return le;
}

static list generic_proper_effects_of_complex_address_point_to_op(list l_args, list *l_pme, int write_p)
{
  /* first get an effect on the structure */
  list le = generic_proper_effects_of_complex_address_expression
    (EXPRESSION(CAR(l_args)), l_pme, write_p);

  /* and add the field */
  FOREACH(EFFECT, pme, *l_pme)
    {
      if(!effect_undefined_p(pme) && !anywhere_effect_p(pme))
	{
	  expression field_exp = EXPRESSION(CAR(CDR(l_args)));
	  syntax s = expression_syntax(field_exp);
	  entity f;
	  effect eff_read;
	  ifdebug(1) pips_assert("e2 is a reference", syntax_reference_p(s));
	  f = reference_variable(syntax_reference(s));
	  /* the pointer is read */
	  pips_debug(4, "we add a read effect on the pointer. \n");

	  eff_read = (*effect_dup_func)(pme);
	  /* memory leak */
	  effect_action(eff_read) = make_action_read_memory();
	  le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));


	  /* We add a dereferencing */
	  effect_add_dereferencing_dimension(pme);

	  /* we add the field dimension */
	  effect_add_field_dimension(pme,f);
	}
    }
  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);
  return le;
}

/* This function should be further cleaned up - see call_to_post_pv - in particular
   pointer arithmetic should be externalized - BC.
*/
static list generic_proper_effects_of_complex_address_dereferencing_op(list l_args, list *l_pme, int write_p)
{
  list le = NIL;
  expression deref_exp = EXPRESSION(CAR(l_args));

  if(expression_call_p(deref_exp))
    {
      call s_c = syntax_call(expression_syntax(deref_exp));
      entity s_op = call_function(s_c);
      list s_args = call_arguments(s_c);
      value op_init = entity_initial(s_op);
      type op_type = ultimate_type(entity_type(s_op));
      tag t = value_tag(op_init);

      pips_debug(4,"The dereferenced expression is a call itself (%s)\n",
		 entity_local_name(s_op));
      ifdebug(8)
	{
	  pips_debug(8,"with arguments : \n");
	  print_expressions(s_args);
	}

      /* MINUS_C should be handled as well BC */
      if(ENTITY_PLUS_C_P(s_op))
	{
	  /* case *(e1+e2) */
	  /* This might be tractable if e1 is a reference to a
	     pointer. For instance, *(p+q-r) can be converted to p[q-r] */
	  expression e1 = EXPRESSION(CAR(s_args));
	  syntax s1 = expression_syntax(e1);
	  expression e2 = EXPRESSION(CAR(CDR(s_args)));
	  expression new_e2 = expression_undefined;

	  pips_debug(8,"*(p+q) case, with p = %s and q = %s\n",
		     words_to_string(words_expression(e1,NIL)),
		     words_to_string(words_expression(e2,NIL)));

	  /* Beware, sometimes, *(p+i+j+k) is represented as *((p+i) + (j+k)):
	     we must first retrieve p
	  */
	  if (syntax_call_p(s1))
	    {
	      call e1_c = syntax_call(s1);
	      entity e1_op = call_function(e1_c);
	      list e1_args = call_arguments(e1_c);

	      if(ENTITY_PLUS_C_P(e1_op) || ENTITY_MINUS_C_P(e1_op))
		{
		  expression e11 = EXPRESSION(CAR(e1_args));
		  expression e22 = EXPRESSION(CAR(CDR(e1_args)));

          /* SG this is a way to handle commutativity of PLUS_C */
          if(ENTITY_PLUS_C_P(e1_op)) {
              basic b1 = basic_of_expression(e1);
              basic b2 = basic_of_expression(e2);
              if(basic_pointer_p(b2)) {
                  expression etmp = e1;
                  e1=e2;
                  e2=etmp;
              }
              free_basic(b1);
              free_basic(b2);
          }


		  pips_debug(8,"p is itself a complicated expression, with e11 = %s and e22 = %s\n",
			     words_to_string(words_expression(e11,NIL)),
			     words_to_string(words_expression(e22,NIL)));
		  /* not too much ;-) */
		  if (syntax_call_p(expression_syntax(e11)))
		    {
		      pips_user_warning("Pips does not currently handle this complicated arithmetic pointer expression\n");
		    }
		  else
		    {
		      e1 = e11;
		      new_e2 = MakeBinaryCall(e1_op, copy_expression(e2), copy_expression(e22));
		      pips_debug(8, "new_e2 : %s\n",
				 words_to_string(words_expression(new_e2,NIL)));
		    }
		}
	    }

	  le = generic_proper_effects_of_complex_address_expression
	    (e1, l_pme, write_p);

	  expression current_e2 = expression_undefined_p(new_e2) ? e2: new_e2;
	  FOREACH(EFFECT, pme, *l_pme)
	    {
	      if (! effect_undefined_p(pme)&& !anywhere_effect_p(pme))
		{
		  syntax s2 = expression_syntax(current_e2);

		  /* we must add a read effect on pme if it is a pointer
		     type
		  */
		  /* deal with case *(p+(i=exp))
		   * the effect is equivalent to an effect on *(p+exp)
		   */
		  if (syntax_call_p(s2))
		    {
		      call s2_c = syntax_call(s2);
		      entity s2_op = call_function(s2_c);
		      list s2_args = call_arguments(s2_c);
		      if (ENTITY_ASSIGN_P(s2_op))
			{
			  current_e2 =  EXPRESSION(CAR(CDR(s2_args)));
			}
		    }

		  if (effect_pointer_type_p(pme))
		    {
		      effect eff_read;
		      pips_debug(4, "It's a pointer; we add a read effect \n");

		      eff_read = (*effect_dup_func)(pme);
		      effect_action(eff_read) = make_action_read_memory();
		      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
		      (*effect_add_expression_dimension_func)(pme, current_e2);
		    }
		  else /* array case */
		    {
		      type pme_t = simple_effect_reference_type(effect_any_reference(pme));

		      if (type_variable_p(pme_t))
			{
			  list l_ind = variable_dimensions(type_variable(pme_t));

			  if (gen_length(l_ind) == 1)
			    {
			      (*effect_add_expression_dimension_func)(pme, current_e2);
			    }
			  else
			    {
			      pips_user_warning("Pips does not precisely handle linearized array references\n");
			      FOREACH(EXPRESSION, ind, l_ind)
				{
				  (*effect_add_expression_dimension_func)(pme, make_unbounded_expression());
				}
			    }
			}
		      else
			{
			  /* replace current effect by an anywhere effect */
			  effect new_eff = make_anywhere_effect
			    (write_p? make_action_write_memory() : make_action_read_memory());
			  free_cell(effect_cell(pme));
			  effect_cell(pme) = effect_cell(new_eff);
			  effect_cell(new_eff) = cell_undefined;
			  free_action(effect_action(pme));
			  effect_action(pme) = effect_action(new_eff);
			  effect_action(new_eff) = action_undefined;
			  free_descriptor(effect_descriptor(pme));
			  effect_descriptor(pme) = effect_descriptor(new_eff);
			  effect_descriptor(new_eff) = descriptor_undefined;
			  effect_to_may_effect(pme);
			  free_effect(new_eff);
			}

		      free_type(pme_t);
		    }
		}
	    }
	  le = gen_nconc(le,
			 generic_proper_effects_of_expression(e2));
	  if (!expression_undefined_p(new_e2))
	    free_expression(new_e2);


	}
      /* Other functions to process: p++, ++p, p--, --p */
      else if(ENTITY_POST_INCREMENT_P(s_op) ||
	      ENTITY_POST_DECREMENT_P(s_op))
	{
	  // case *(e1++) or *(e1--)
	  // the is a read and write effect on e1
	  expression e1 = EXPRESSION(CAR(s_args));

	  le = generic_proper_effects_of_complex_address_expression
	    (e1, l_pme, write_p);

	  FOREACH(EFFECT, pme, *l_pme)
	    {
	      if (! effect_undefined_p(pme)&& !anywhere_effect_p(pme))
		{
		  /* we must add a read effect on *pme if it is a pointer
		     type
		  */
		  if (effect_pointer_type_p(pme))
		    {
		      pips_debug(4, "It's a pointer; we add a read and write effect \n");

		      effect eff_read = (*effect_dup_func)(pme);
		      effect_action(eff_read) = make_action_read_memory();
		      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
		      effect eff_write = (*effect_dup_func)(pme);
		      effect_action(eff_write) = make_action_write_memory();
		      le = gen_nconc(le, CONS(EFFECT, eff_write, NIL));
		    }
		  /* We add a dereferencing */
		  effect_add_dereferencing_dimension(pme);
		}
	    }
	}
      else if(ENTITY_PRE_INCREMENT_P(s_op) ||
	      ENTITY_PRE_DECREMENT_P(s_op))
	{
	  expression e1 = EXPRESSION(CAR(s_args));
	  syntax s1 = expression_syntax(e1);
	  reference r1 = syntax_reference(s1);
	  reference nr1 = reference_undefined;

	  /* YOU DO NOT WANT TO GO DOWN RECURSIVELY. DO AS FOR C_PLUS ABOVE */

	  pips_assert("The argument is a reference", syntax_reference_p(s1));
	  pips_assert("The reference is scalar", ENDP(reference_indices(r1)));

	  le = generic_proper_effects_of_expression(EXPRESSION(CAR(l_args)));
	  nr1 = copy_reference(r1);
	  if(ENTITY_PRE_INCREMENT_P(s_op))
	    reference_indices(nr1) = CONS(EXPRESSION, int_to_expression(1), NIL);
	  else
	    reference_indices(nr1) = CONS(EXPRESSION, int_to_expression(-1), NIL);

	  /* Too bad for the memory leaks involved... This deref_exp
	     should be freed at exit. */
	  *l_pme = effect_to_list((*reference_to_effect_func)
				  (nr1, write_p? make_action_write_memory() : make_action_read_memory(), false));
	}
      else if(ENTITY_ADDRESS_OF_P(s_op))
	{
	  // case *&a
	  // this is an effect on a
	  expression e1 = EXPRESSION(CAR(s_args));
	  le = generic_proper_effects_of_complex_address_expression(e1,
								    l_pme,
								    write_p);
	}
      else
	{
	  /* do nothing, go down recursively to handle other calls */
	  pips_debug(8, "We go down recursively\n");
	  le = generic_proper_effects_of_complex_address_expression
	    (deref_exp, l_pme, write_p);

	  bool success = !ENDP(*l_pme);
	  if (!success)
	    {
	      if (type_functional_p(op_type) && t == is_value_constant)
		{
		  constant op_const = value_constant(op_init);
		  if (constant_int_p(op_const) && (constant_int(op_const) == 0))
		    {
		      success = false;
		    }
		  else
		    {
		      type tt = functional_result(type_functional(op_type));
		      if (type_variable_p(tt))
			{
			  variable v = type_variable(tt);
			  basic b = variable_basic(v);
			  if (basic_string_p(b))/* constant strings */
			    {
			      /*the user dereferences a constant string -> it's a constant,
			       -> no effect */
			      success = true;
			    }
			}
		    }
		}
	    }
	  if (!success)
	    {
	      pips_user_warning("dereferencing a constant address expression: "
				"PIPS doesn't know how to handled that precisely\n");
	      *l_pme = effect_to_list(make_anywhere_effect(write_p?
							   make_action_write_memory()
							   : make_action_read_memory()));
	    }
	  else
	    {
	      FOREACH(EFFECT, pme, *l_pme)
		{
		  if(!effect_undefined_p(pme) && !anywhere_effect_p(pme))
		    {
		      effect eff_read;
		      expression e1 = EXPRESSION(CAR(l_args));
		      type e1t = expression_to_type(e1);

		      pips_debug(4,"adding a read effect on dereferenced expression\n");

		      /* we add the read effect on the dereferenced expression
			 if it's a pointer
		      */
		      if (type_variable_p(e1t) &&
			  basic_pointer_p(variable_basic(type_variable(e1t)))
			  && ENDP(variable_dimensions(type_variable(e1t))))
			{
			  pips_debug(8, "adding read effect on argument \n");
			  eff_read = (*effect_dup_func)(pme);
			  /* memory leak? */
			  effect_action_(eff_read) = make_action_read_memory();
			  le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
			}

		      free_type (e1t);

		      pips_debug(8, "adding dereferencing dimension \n");
		      effect_add_dereferencing_dimension(pme);
		    }
		}
	    }

	}
    }
  else if (syntax_va_arg_p(expression_syntax(deref_exp)))
    {
      /* there could be more work here, but va_arg is very poorly
	 handled everywhere for the moment.
      */
      /* we generated an effect on the va_list, and that is all */
      list vararg_list = syntax_va_arg(expression_syntax(deref_exp));
      sizeofexpression soe = SIZEOFEXPRESSION(CAR(vararg_list));

      pips_debug(4,"The dereferenced expression is a va_arg\n");

      le = generic_proper_effects_of_complex_address_expression(sizeofexpression_expression(soe), l_pme, true);
      /* and we must add an anywhere effect because we don't know where
	 the dereferenced location is. TO BE CHECKED OR REFINED
      */
      le = CONS(EFFECT,
		make_anywhere_effect
		(write_p? make_action_write_memory() : make_action_read_memory()),
		le);
    }
  else
    {
      /* This is not a call, go down recursively */
      pips_debug(4,"The dereferenced expression is not a call itself : we go down recursively\n");
      le = generic_proper_effects_of_complex_address_expression
	(deref_exp, l_pme, write_p);

      if (ENDP(*l_pme))
	{
	  pips_user_warning("dereferencing a constant address expression: "
			    "PIPS doesn't know how to handle that precisely\n");
	  *l_pme = effect_to_list(make_anywhere_effect(write_p? make_action_write_memory()
						       : make_action_read_memory()));
	}
      else
	{
	  FOREACH(EFFECT, pme, *l_pme)
	    {
	      if(!effect_undefined_p(pme) && !anywhere_effect_p(pme))
		{
		  effect eff_read;
		  expression e1 = EXPRESSION(CAR(l_args));
		  type e1t = expression_to_type(e1);

		  pips_debug(4,"adding a read effect on dereferenced expression\n");

		  /* we add the read effect on the dereferenced expression
		     if it's a pointer
		  */
		  if (type_variable_p(e1t) &&
		      basic_pointer_p(variable_basic(type_variable(e1t)))
		      && ENDP(variable_dimensions(type_variable(e1t))))
		    {
		      pips_debug(8, "adding read effect on argument \n");
		      eff_read = (*effect_dup_func)(pme);
		      /* memory leak? */
		      effect_action_(eff_read) = make_action_read_memory();
		      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
		    }

		  free_type (e1t);

		  pips_debug(8, "adding dereferencing dimension \n");
		  effect_add_dereferencing_dimension(pme);
		}
	    }
	}

    }

  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "End with *l_pme=\n", *l_pme);

  return le;
}


static list generic_proper_effects_of_complex_address_call_expression(expression call_exp, list *l_pme, int write_p)
{
  list le = NIL;
  call c = syntax_call(expression_syntax(call_exp));
  entity op = call_function(c);
  list args = call_arguments(c);
  /* FI: we assume there it at least one argument */
  pips_debug(4, "This is a call\n");

  *l_pme = NIL;

  if(ENTITY_FIELD_P(op))
    {
      pips_debug(4, "Call is a field operator\n");
      le = generic_proper_effects_of_complex_address_field_op(args, l_pme, write_p);
    }
  else if (ENTITY_POINT_TO_P(op))
    {
      pips_debug(4, "Call is a point to operator\n");
      le = generic_proper_effects_of_complex_address_point_to_op(args, l_pme, write_p);
    }
  else if(ENTITY_DEREFERENCING_P(op))
    {
      pips_debug(4, "Call is a dereferencing operator \n");
      syntax op_arg_s = expression_syntax(EXPRESSION(CAR(args)));
      if (syntax_cast_p(op_arg_s))
	{
	  pips_debug(4, "Dereferencing a cast expression \n");
	  // we cannot call generic_proper_effects_of_complex_address_dereferencing_op here
	  // because we need to test the relationship between the type of call_exp and cast_exp.

	  cast c = syntax_cast(op_arg_s);
	  expression cast_exp = cast_expression(c);

	  // try to see if we have something like *((int (*)[]) t) where t is of type int * for instance
	  type call_exp_t = expression_to_type(call_exp);
	  type cast_exp_t = expression_to_type(cast_exp);

	  list l_me1 = NIL;
	  le = generic_proper_effects_of_complex_address_expression(cast_exp, &l_me1, write_p);
	  // re-use an existing function because currently it checks if types are the same
	  // or if each array dimension corresponds to a pointer dimension
	  if (!ENDP(l_me1) && !anywhere_effect_p(EFFECT(CAR(l_me1))))
	    {
	      if (types_compatible_for_effects_interprocedural_translation_p(call_exp_t,  cast_exp_t))
		{
		  // current result is ok;
		  pips_debug(4, "compatible types \n");
		  *l_pme = l_me1;

		  if (write_p && pointer_type_p(cast_exp_t))
		    {
		      FOREACH(EFFECT, eff, l_me1)
			{
			  effect read_eff = (*effect_dup_func)(eff);
			  effect_to_read_effect(read_eff);
			  le = CONS(EFFECT, read_eff, le);
			}
		    }

		}
	      else
		{
		  pips_debug(4, "non compatible types \n");

		  // we generate all possible paths from the cast expression main effects
		  /* No, we currently don't know how to handle that at all call sites */
		  /* It may be more advisable to return an effect on the base adress, and check the
		     type of the base adress. If it's not compatible, then all paths effects
		     should be generated if it is appropriate.
		  */
		  /* FOREACH(EFFECT, eff, l_me1) */
/* 		    { */
/* 		      list l_tmp =  generic_effect_generate_all_accessible_paths_effects(eff, cast_t, write_p?'w':'r'); */
/* 		      effects_to_may_effects(l_tmp); */
/* 		      *l_pme = gen_nconc(l_tmp, *l_pme); */
/* 		    } */

		  pips_user_warning("PIPS currently does not know how to precisely handle "
				    "complex cast expressions\n");
		  *l_pme = effect_to_list(make_anywhere_effect
					  (write_p ? make_action_write_memory() : make_action_read_memory()));
		}
	    }
	  else
	    {
	      pips_user_warning("dereferencing a constant address expression: "
				"PIPS doesn't know how to handled that precisely\n");
	      *l_pme = effect_to_list(make_anywhere_effect(write_p?
							   make_action_write_memory()
							   : make_action_read_memory()));
	    }
	  free_type(call_exp_t);
	  free_type(cast_exp_t);
	}
      else
	{
	  le = generic_proper_effects_of_complex_address_dereferencing_op(args, l_pme, write_p);
	}
    }
  else if(ENTITY_CONDITIONAL_P(op))
    {
      pips_debug(4, "Call is a conditional operator\n");
      list l_pme1 = NIL;
      list l_pme2 = NIL;
      expression cond = EXPRESSION(CAR(args));
      expression e1 = EXPRESSION(CAR(CDR(args)));
      expression e2 = EXPRESSION(CAR(CDR(CDR(args))));

      list le1 = generic_proper_effects_of_complex_address_expression(e1, &l_pme1, write_p);
      effects_to_may_effects(le1);
      effects_to_may_effects(l_pme1);
      list le2 = generic_proper_effects_of_complex_address_expression(e2, &l_pme2, write_p);
      effects_to_may_effects(le2);
      effects_to_may_effects(l_pme2);
      *l_pme = gen_nconc(l_pme1, l_pme2);
      le = gen_nconc(le1, le2);
      le = gen_nconc(generic_proper_effects_of_expression(cond), le);
    }
  else
    {
      /* failure: a user function is called to return a structure or an address */
      /* if it's a structure, it's ok -> no effect. If it's an adress, it may be anywhere
         until we can retrieve it from pointer analysis
      */
      type t = expression_to_type(call_exp);
      if(pointer_type_p(t) || array_type_p(t))
	{
	  pips_user_warning("PIPS currently does not know how to precisely handle "
			    "address values used in complex call expressions expression\n");
	  *l_pme = effect_to_list(make_anywhere_effect
			      (write_p ? make_action_write_memory() : make_action_read_memory()));
	}
      else
	{
	  pips_debug(8, "constant return value: -> no main effects\n");
	}
      free_type(t);
      le = generic_proper_effects_of_expression(call_exp);
    }

  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);
  return le;
}

static list generic_proper_effects_of_complex_address_cast_expression(cast c, list *l_pme, int write_p)
{
  list le = NIL;

  /* FI: The cast has an impact on pointer arithmetic. I do not know
     how to take it into account. The cast may also use a typedef
     type. */
  /* BC : it is a translation. The main effect should be translated
     accordingly to the cast type on return of the recursion. */
  pips_user_warning("Cast impact on pointer arithmetic and indexing is ignored\n");
  if(!get_bool_property("MEMORY_EFFECTS_ONLY")) {
    type ct = cast_type(c);

    if(typedef_type_p(ct)) {
      entity te = basic_typedef(variable_basic(type_variable(ct)));
      effect tre = make_declaration_effect(te, false); // type
      le = gen_nconc(le, CONS(EFFECT, tre, NIL));
    }
  }
  expression cast_exp = cast_expression(c);
  pips_debug(8, "We go down recursively on the cast expression \n");
  le = gen_nconc(le, generic_proper_effects_of_complex_address_expression
		 (cast_exp, l_pme, write_p));

  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);


  return le;
}

static list generic_proper_effects_of_complex_address_subscript_expression(subscript subsc, list *l_pme, int write_p)
{
  pips_debug(8, "begin\n");

  list le = NIL;

  expression subsc_exp = subscript_array(subsc);
  list ind = subscript_indices(subsc);
  type t_subsc_exp = expression_to_type(subsc_exp);
  pips_debug(8, "We go down recursively on the subscripted expression \n");
  le = generic_proper_effects_of_complex_address_expression
    (subsc_exp, l_pme, write_p);

  FOREACH(EFFECT, pme, *l_pme)
    {
      if(!effect_undefined_p(pme) && !anywhere_effect_p(pme))
	{
	  /* if the array expression is a pointer, we must add a read
	     effect on it, that is to say a read effect on pme.
	  */
	  if (pointer_type_p(t_subsc_exp))
	    {
	      effect eff_read = (*effect_dup_func)(pme);

	      pips_debug(5, "adding read effect on array expression\n");
	      /* memory leak? */
	      effect_action(eff_read) = make_action_read_memory();
	      le = gen_nconc(le, CONS(EFFECT, eff_read, NIL));
	    }

	  /* We add the corresponding dimensions to the effect *pme
	   * and we should add read effects on pointer subscript indices
	   */
	  FOREACH(EXPRESSION, ind_exp, ind)
	    {
	      (*effect_add_expression_dimension_func)(pme, ind_exp);
	      le = gen_nconc(le, generic_proper_effects_of_expression(ind_exp));

	    }
	}
    }
  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);

  return le;
}

static list generic_proper_effects_of_complex_address_va_arg_expression(list __attribute__((unused)) v, list *l_pme, int write_p)
{
  pips_debug(8, "begin");

  list le = NIL;

  /* The built-in can return a pointer which is dereferenced */
  /* va_args is read... */
  *l_pme = effect_to_list(make_anywhere_effect
			  (write_p ? make_action_write_memory() : make_action_read_memory()));

  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);

  return le;
}



/**
 @return : a list of read effects corresponding to intermediate read memory
         accesses during the evaluation of add_exp.
 @param add_exp is the expression which memory effects we are looking for
 @param pme is a Pointer towards the Main memory Effect of add_exp.
 @param write_p is a boolean set to true if the main effect is write, false
         otherwise.

 Go down along the first argument till you find a reference or a
 dereferencing and build the effect *pme by side effects as well as the
 auxiliary effect list le on the way back up.
 checks at each step that no effect is generated on partially subscripted
 arrays.
*/
list generic_proper_effects_of_complex_address_expression(expression add_exp, list *l_pme, int write_p)
{
  list le = NIL;
  syntax s = expression_syntax(add_exp);

  pips_debug(3, "begin for expression : %s\n",
	     words_to_string(words_expression(add_exp,NIL)));

  if(syntax_reference_p(s))
    {
      pips_debug(4, "reference case \n");
      reference ref = syntax_reference(s);
      effect pme;
      le = generic_p_proper_effect_of_reference(ref, &pme,
						write_p, true);
      if (pme != effect_undefined)
	*l_pme = effect_to_list(pme);
      else
	*l_pme = effect_to_list(make_anywhere_effect(write_p?make_action_write_memory(): make_action_read_memory()));

    }
  else if(syntax_call_p(s))
    {
      pips_debug(4, "call case\n");
      le = generic_proper_effects_of_complex_address_call_expression(add_exp, l_pme, write_p);
    }
  else if(syntax_cast_p(s))
    {
      pips_debug(4, "cast case\n");
      le = generic_proper_effects_of_complex_address_cast_expression(syntax_cast(s), l_pme, write_p);

    }
  else if(syntax_subscript_p(s))
    {
      pips_debug(4,"subscript case\n");
      le = generic_proper_effects_of_complex_address_subscript_expression(syntax_subscript(s), l_pme, write_p);
    }
  else if(syntax_va_arg_p(s))
    {
      pips_debug(4,"va_arg case\n");
      le = generic_proper_effects_of_complex_address_va_arg_expression(syntax_va_arg(s), l_pme, write_p);
    }
  else
    {
      /* sizeofexpression, application.*/
      pips_internal_error("Unexpected case");
    }

  pips_debug_effects(8, "End with le=\n", le);
  pips_debug_effects(8, "and with *l_pme=\n", *l_pme);

  return le;
}


list generic_proper_effects_of_any_lhs(expression lhs)
{
  return generic_proper_effects_of_address_expression(lhs, true);
}


/**
 @return : a list of read effects corresponding to intermediate read memory
         accesses during the evaluation of add_exp.
 @param add_exp is the expression which memory effects we are looking for
 @param l_addexp_pme is a Pointer towards the Main memory Effects of add_exp.
 @param lpme is a pointer towars the memory effects of the expression if it is
        of type struct or union (but not an array of structs or union).
 @param write_p is a boolean set to true if the main effect is write, false
         otherwise.

 This function is an interface to generic_proper_effects_of_complex_address_expression
 that also generates effects on paths accessible from struct and union fields down
 to pointers if the expression type is a struct or union (basic derived but not enum, and
 no dimensions).
 If *lpme is not set to NIL, then the main effects have no real sense (effects on a variable name),
 and should be freed.
*/
list generic_proper_effects_of_complex_memory_access_expression(expression addexp, list *l_addexp_pme, list *lpme, int write_p)
{
  list le = NIL;
  *lpme = NIL;

  pips_debug(5, "call or subscript case\n");
  /* Look for a main read-write effect of the lhs and for its
     secondary effects */
  le = generic_proper_effects_of_complex_address_expression(addexp, l_addexp_pme, write_p);


  FOREACH(EFFECT, pme, *l_addexp_pme)
    {
      if(!effect_undefined_p(pme))
	{

	  type addexp_t = expression_to_type(addexp);

	  if (type_variable_p(addexp_t))
	    {
	      variable addexp_tv = type_variable(addexp_t);
	      if (ENDP(variable_dimensions(addexp_tv)))
		{
		  basic addexp_tvb = variable_basic(addexp_tv);
		  if (basic_derived_p(addexp_tvb) && !(type_enum_p(entity_type(basic_derived(addexp_tvb)))))
		    {
		      *lpme = gen_nconc(generic_r_proper_effects_of_derived_reference(pme, addexp_t), *lpme);
		    }
		}
	      else
		{
		  pips_debug(8, "main effect is on array name \n");

		}
	    }
	  else
	    {
	      pips_internal_error("case not handled yet ");
	    }

	  free_type(addexp_t);


	}
    }
  transformer context = transformer_undefined;
  if( effects_private_current_context_stack_initialized_p()
      && !effects_private_current_context_empty_p())
    context = effects_private_current_context_head();
  if (!transformer_undefined_p(context))
    (*effects_precondition_composition_op)(*lpme, context);

  return le;

}

list generic_proper_effects_of_address_expression(expression addexp, int write_p)
{
  list le = NIL;
  syntax s = expression_syntax(addexp);


  pips_debug(5, "begin for expression : %s\n",
	     words_to_string(words_expression(addexp,NIL)));

  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      {
	pips_debug(5, "reference case\n");
	if(write_p)
	  le = generic_proper_effects_of_written_reference(syntax_reference(s));
	else
	  pips_internal_error("Case not taken into account");
	break;
      }
    case is_syntax_call:
      pips_debug(5, "call case\n");
    case is_syntax_subscript:
      {
	list l_pme = NIL; /* main data read-write effect: p[*] */

	pips_debug(5, "call or subscript case\n");
	/* Look for a main read-write effect of the lhs and for its
	   secondary effects */
	le = generic_proper_effects_of_complex_address_expression(addexp, &l_pme, write_p);

	FOREACH(EFFECT, e, l_pme)
	  {
	    if(!effect_undefined_p(e))
	      {
		transformer context = transformer_undefined;

		if( effects_private_current_context_stack_initialized_p()
		    && !effects_private_current_context_empty_p())
		  context = effects_private_current_context_head();

		type addexp_t = expression_to_type(addexp);

		/* we add the read effect if it's not an array name
		 */
		if (type_variable_p(addexp_t))
		  {
		    variable addexp_tv = type_variable(addexp_t);
		    if (ENDP(variable_dimensions(addexp_tv)))
		      {
			basic addexp_tvb = variable_basic(addexp_tv);
			pips_debug(8, "adding main read effect \n");
			if (basic_derived_p(addexp_tvb) && !(type_enum_p(entity_type(basic_derived(addexp_tvb)))))
			  {
			    list l_tmp = generic_r_proper_effects_of_derived_reference(e, addexp_t);
			    le = gen_nconc(le, l_tmp);
			  }
			else
			  le = CONS(EFFECT, e, le);
		      }
		    else
		      {
			pips_debug(8, "main read effect is on array name : discarded\n");
			free_effect(e);
		      }
		  }
		else if (type_functional_p(addexp_t))
		  {
		    // FI: we must have bumped into a pointer to a
		    // function. It may be dereferenced or not. See
		    // Effects-New/function03
		    print_expression(addexp);
		    pips_internal_error("case of address expression not handled yet\n");
		  }
		else
		  {
		    pips_internal_error("case not handled yet ");
		  }

		free_type(addexp_t);
		if (!transformer_undefined_p(context))
		  (*effects_precondition_composition_op)(le, context);

	      }
	  }
	gen_free_list(l_pme);

	ifdebug(8) {
	  pips_debug(8, "Effect for a call or a subscripted expression:\n");
	  (*effects_prettyprint_func)(le);
	}
	break;
      }
    case is_syntax_cast:
      {
	pips_user_error("use of cast expressions as lvalues is deprecated\n");
	break;
      }
    case is_syntax_sizeofexpression:
      {
	pips_user_error("sizeof cannot be a lhs\n");
	break;
      }
    case is_syntax_application:
      {
	/* I assume this not any more possible than a standard call */
	pips_user_error
	  ("use of indirect function call as lhs is not allowed\n");
	break;
      }
    default:
      pips_internal_error
	("lhs is not a reference and is not handled yet: syntax tag=%d\n",
	 syntax_tag(s));

    } /* end switch */


  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
    fprintf(stderr, "\n");
  }

  return le;
}


/* TO VERIFY !!!!!!!!!!!!!*/
/* UNUSED ? */
list
generic_proper_effects_of_subscript(subscript s)
{
    list inds = subscript_indices(s);
    list le = NIL;
    transformer context;

    if (!effects_private_current_context_stack_initialized_p()
	|| effects_private_current_context_empty_p())
	context = transformer_undefined;
    else
      {
	context = effects_private_current_context_head();
      }


    pips_debug(3, "begin\n");

    if (! (*empty_context_test)(context))
    {
      le = generic_proper_effects_of_expression(subscript_array(s));

      if (! ENDP(inds))
	le = gen_nconc(le, generic_proper_effects_of_expressions(inds));


	(*effects_precondition_composition_op)(le, context);
    }

    pips_debug(3, "end\n");
    return(le);
}

list generic_proper_effects_of_application(application a __attribute__((__unused__)))
{
  list le = NIL;

  pips_user_warning("Effect of indirect calls not implemented -> returning anywhere\n");

  le = make_anywhere_read_write_memory_effects();
  return(le);
}



/* Compute the proper effects of an expression

   @param[in] e is the expression we want the effects

   @return the corresponding list of effects.

   It calls store_expr_prw_effects() to keep track of expression effects
   if needed
*/
list
generic_proper_effects_of_expression(expression e)
{
  list le = NIL;
  syntax s;

  pips_debug(3, "begin\n");

  s = expression_syntax(e);

  switch(syntax_tag(s))
    {
    case is_syntax_reference:
      le = generic_proper_effects_of_read_reference(syntax_reference(s));
      break;
    case is_syntax_range:
      le = generic_proper_effects_of_range(syntax_range(s));
      break;
    case is_syntax_call:
      {
	entity op = call_function(syntax_call(s));

	/* first the case of an adressing operator : this could also be done
	 * by calling generic_r_proper_effects_of_call, but then the expression
	 * is lost and must be rebuild later to call
	 * g_p_e_of_address_expression.
	 */
	if (ENTITY_FIELD_P(op) ||
	    ENTITY_POINT_TO_P(op) ||
	    ENTITY_DEREFERENCING_P(op))
	  le = generic_proper_effects_of_address_expression(e, false);
	else {
	  le = generic_r_proper_effects_of_call(syntax_call(s));
	  if(entity_variable_p(op)) {
	    /* op is a pointer to a function. A memory read effect
	       must be added as this is in fact a read reference. See
	       Effects-new/function03.c */
	    reference r = make_reference(op, NIL);
	    list ople = generic_proper_effects_of_read_reference(r);
	    // FI: by default, a preference will be used. This
	    //artifical reference cannot be freed. I whish there would
	    //be a cleaner way to do this:
	    // generic_proper_effects_of_read_entity()?
	    //free_reference(r);
	    le = gen_nconc(ople, le);
	  }
	}
	break;
      }
    case is_syntax_cast:
      {
	le = generic_proper_effects_of_expression(cast_expression(syntax_cast(s)));
	if(!get_bool_property("MEMORY_EFFECTS_ONLY")) {
	  type ct = cast_type(syntax_cast(s));

	  if(typedef_type_p(ct)) {
	    entity te = basic_typedef(variable_basic(type_variable(ct)));
	    effect tre = make_declaration_effect(te, false); // type
	    le = gen_nconc(le, CONS(EFFECT, tre, NIL));
	  }
	}

	break;
      }
    case is_syntax_sizeofexpression:
      {
	sizeofexpression se = syntax_sizeofexpression(s);
	if (sizeofexpression_expression_p(se))
	  {
	    /* FI: If the type of the reference is a dependent type, this
	       may imply the reading of some expressions... See for
	       instance type_supporting_entities()? Is sizeof(a[i]) ok? */
	    /* The type of the variable is read, not the variable itself.*/
	    /* le = generic_proper_effects_of_expression(sizeofexpression_expression(se)); */
	  ;
	  }
	else {
	  if(!get_bool_property("MEMORY_EFFECTS_ONLY")) {
	    type sot = sizeofexpression_type(se);

	    if(typedef_type_p(sot)) {
	      entity te = basic_typedef(variable_basic(type_variable(sot)));
	      effect tre = make_declaration_effect(te, false); // type
	      le = gen_nconc(le, CONS(EFFECT, tre, NIL));
	    }
	  }
	}
	break;
      }
    case is_syntax_subscript:
      {
	le = generic_proper_effects_of_address_expression(e, false);
	break;
      }
    case is_syntax_application:
      le = generic_proper_effects_of_application(syntax_application(s));
      break;
    case is_syntax_va_arg:
      {
	/* there is first a read of the first argument, and
	   subsequent write effects on the va_list depths are simulated
	   by write effects on the va_list itself.
	*/
	list al = syntax_va_arg(s);
	expression ae = sizeofexpression_expression(SIZEOFEXPRESSION(CAR(al)));
	le = generic_proper_effects_of_expression(ae);
	le = gen_nconc(le, generic_proper_effects_of_any_lhs(ae));
	break;
      }
    default:
      pips_internal_error("unexpected tag %d", syntax_tag(s));
    }

  ifdebug(8)
    {
	pips_debug(8, "Proper effects of expression \"%s\":\n",
		   words_to_string(words_syntax(s,NIL)));
	(*effects_prettyprint_func)(le);
    }

  /* keep track of proper effects associated to sub-expressions if required.
   */
  if (!expr_prw_effects_undefined_p())
    {
      /* in IO lists, the effects are computed twice,
     * once as LHS, once as a REFERENCE...
     * so something may already be in. Let's skip it.
     * I should investigate further maybe. FC.
     */
      if (!bound_expr_prw_effects_p(e))
	store_expr_prw_effects(e, make_effects(gen_full_copy_list(le)));
    }

  return le;
}

/* list generic_proper_effects_of_expressions(list exprs)
 * input    : a list of expressions and the current context.
 * outpproper_ut   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :
 */
list
generic_proper_effects_of_expressions(list exprs)
{
    list le = NIL;

    pips_debug(5, "begin\n");

    MAP(EXPRESSION, exp,
	/* le may be long... */
	le = gen_nconc(generic_proper_effects_of_expression(exp), le),
	exprs);

    pips_debug(5, "end\n");
    return(le);
}

bool check_sdfi_effects_p(entity func, list func_sdfi)
{
  list ce = list_undefined;
  bool check_p = true;
  type ut = ultimate_type(entity_type(func));

  pips_assert("func is a function", type_functional_p(ut));

  /* Check the SDFI effects */
  for(ce = func_sdfi; !ENDP(ce); POP(ce)) {
    effect eff = EFFECT(CAR(ce));
    reference r = effect_any_reference(eff);
    entity v = reference_variable(r);

    if(formal_parameter_p(v)) {
      storage s = entity_storage(v);
      formal fs = storage_formal(s);
      int rank = formal_offset(fs);
      entity called_function = formal_function(fs);

      if(called_function!=func) {
	fprintf(stderr, "Summary effect %p for function \"%s\" refers to "
		"formal parameter \"%s\" of function \"%s\"\n",
		eff, entity_name(func), entity_name(v), entity_name(called_function));
	check_p = false;
      }

      if(rank> (int) gen_length(functional_parameters(type_functional(ut)))) {
	fprintf(stderr, "Formal parameter \"%s\" is ranked %d out of %zd!\n",
		entity_name(v), rank, gen_length(functional_parameters(type_functional(ut))));
	check_p = false;
      }
    }
  }
  return check_p;
}

static list
generic_proper_effects_of_external(entity func, list args)
{
    list le = NIL;
    const char *func_name = module_local_name(func);

    pips_debug(4, "translating effects for %s\n", func_name);

    if (! entity_module_p(func))
    {
	pips_internal_error("%s: bad function", func_name);
    }
    else
    {
	list func_eff;
	transformer context;

        /* Get the summary effects of "func". */
	func_eff = (*db_get_summary_rw_effects_func)(func_name);

	if(!check_sdfi_effects_p(func, func_eff))
	  pips_internal_error("SDFI effects for \"%s\" are corrupted in the data base",
			      entity_name(func));

	/* Translate them using context information. */
	context = effects_private_current_context_head();
	le = generic_effects_backward_translation(func, args, func_eff, context);

	if(!check_sdfi_effects_p(func, func_eff))
	  pips_internal_error("SDFI effects for \"%s\" have been corrupted by the translation",
			      entity_name(func));
    }
    ifdebug(1) pips_assert("All effects in \"le\" are consistent",
			     effect_list_consistent_p(le));
    return le;
}

list generic_proper_effects_of_c_function_call_argument(expression arg)
{
  list le = NIL;
  type arg_t = type_undefined;
  pips_debug(3, "begin for actual argument: %s\n", words_to_string(words_expression(arg, NIL)));

  arg_t = expression_to_type(arg);

  /* If it's a sub-array, there are only intermediate effects */
  if(type_variable_p(arg_t) && !ENDP(variable_dimensions(type_variable(arg_t))))
    {
      list l_pme = NIL;
      /* I'm not sure it is OK for all type of arguments, in particular function calls */
      le = generic_proper_effects_of_complex_address_expression(arg, &l_pme, false);

      // when there are casts, actual types are hidden
      if (!ENDP(l_pme))
	{
	  effect eff = EFFECT(CAR(l_pme));
	  reference ref = effect_any_reference(eff);


	  if (ENDP(reference_indices(ref))
	      && pointer_type_p(entity_basic_concrete_type(reference_variable(ref))))
	    {
	      le = gen_nconc(le, l_pme);
	    }
	  else
	    gen_full_free_list(l_pme);

	}
    }
  else
    {
      le = generic_proper_effects_of_expression(arg);
    }


  pips_debug_effects(3, "ouput effects: \n", le);

  return le;
}


/**

 * @return the list of effects found.
 * @param c, a call, which can be a call to a subroutine, but also
 * to an function, or to an intrinsic, or even an assignement.
 */
list
generic_r_proper_effects_of_call(call c)
{
  list le = NIL;
  entity e = call_function(c);
  tag t = value_tag(entity_initial(e));
  const char* n = module_local_name(e);
  list pc = call_arguments(c);
  type uet = ultimate_type(entity_type(e));

  pips_debug(2, "begin for %s\n", entity_local_name(e));

  if(type_functional_p(uet)) {
    switch (t) {
    case is_value_code:
      pips_debug(5, "external function %s\n", n);
      le = generic_proper_effects_of_external(e, pc);
      break;

    case is_value_intrinsic:
      pips_debug(5, "intrinsic function %s\n", n);
      le = generic_proper_effects_of_intrinsic(e, pc);
      break;

    case is_value_symbolic:
      pips_debug(5, "symbolic\n");
      break;

    case is_value_constant:
      pips_debug(5, "constant\n");
      break;

    case is_value_unknown:
      if (get_bool_property("HPFC_FILTER_CALLEES"))
	/* hpfc specials are managed here... */
	le = NIL;
      else
	pips_internal_error("unknown function %s", entity_name(e));
      break;

    default:
      pips_internal_error("unknown tag %d", t);
      break;
    }
  }
  else if(type_variable_p(uet)) {
    /* We could be less optimistic even when no information about the function called is known.
     *
     * We could look up all functions with the same type and make the union of their effects.
     *
     * We could assume that all parameters are read.
     *
     * We could assume that all pointers are used to produce indirect write.
     */
    pips_user_warning("Effects of call thru functional pointers are ignored\n");
  }
  else if(type_statement_p(uet)) {
    le = NIL;
  }
  else {
    pips_internal_error("Unexpected case");
  }

  pips_debug(2, "end\n");

  return(le);
}


/**************************************************************** STATEMENTS */

static void
proper_effects_of_call(call c)
{
    list l_proper = NIL;
    statement current_stat = effects_private_current_stmt_head();
    instruction inst = statement_instruction(current_stat);

    /* Is the call an instruction, or a sub-expression? */
    if (instruction_call_p(inst) && (instruction_call(inst) == c))
    {
      pips_debug(2, "Effects for statement %03zd:\n",
		   statement_ordering(current_stat));

	l_proper = generic_r_proper_effects_of_call(c);

	l_proper = gen_nconc(l_proper, effects_dup(cumu_range_effects()));

	if (contract_p)
	    l_proper = proper_effects_contract(l_proper);

	ifdebug(2) {
	  pips_debug(2, "Proper effects for statement %03zd:\n",
		     statement_ordering(current_stat));
	  (*effects_prettyprint_func)(l_proper);
	  pips_debug(2, "end\n");
	}

	/* This change is not compatible with previous Fortran
	   oriented phases such as hpfc. We can either forbid this
	   feature for all Fortran codes, or add a new property to
	   control this effect elimination, specific to C but with an
	   impact on CONTINUE, for instance, in Fortran. */
	if(!ENDP(l_proper)
	   && effects_all_read_p(l_proper)
	   && !statement_may_have_control_effects_p(current_stat)
	   && !format_statement_p(current_stat)
	   && !fortran_module_p(get_current_module_entity())) {
	  /* The current statement should be ignored as it does not
	     impact the store, nor the control, nor the
	     formatting. Examples in C; "0;" or "i;" or "(void)
	     i". Because PIPS is interprocedural, it could ignore some
	     more statements than gcc, but control effects are not
	     analyzed. Such statements can be created by program
	     transformations. */
	  if (!declaration_statement_p(current_stat))
	    pips_user_warning("Statement %d is ignored because it does not "
			      "modify the store.\n", statement_number(current_stat));
	  gen_full_free_list(l_proper);
	  l_proper = NIL;
	}

	if (get_constant_paths_p())
	  {
	    list l_tmp = l_proper;
	    l_proper = pointer_effects_to_constant_path_effects(l_proper);
	    effects_free(l_tmp);
	  }

	store_proper_rw_effects_list(current_stat, l_proper);
    }
}

/* just to handle one kind of instruction, expressions which are not calls */
static void proper_effects_of_expression_instruction(instruction i)
{
  list l_proper = NIL;
  statement current_stat = effects_private_current_stmt_head();
  //instruction inst = statement_instruction(current_stat);

  /* Is the call an instruction, or a sub-expression? */
  if (instruction_expression_p(i)) {
    expression ie = instruction_expression(i);
    syntax is = expression_syntax(ie);
    call c = call_undefined;

    switch (syntax_tag(is))
    {
    case is_syntax_cast :
      {
	expression ce = cast_expression(syntax_cast(is));
	syntax sc = expression_syntax(ce);

	if(syntax_call_p(sc)) {
	  c = syntax_call(sc);
	  l_proper = generic_r_proper_effects_of_call(c);
	}
	else if(syntax_reference_p(sc)) {
	  /* FI: I guess you do not end up here if the cast appears in
	     the lhs, assuming this is till compatible with the
	     standard. */
	  reference r = syntax_reference(sc);
	  l_proper = generic_proper_effects_of_read_reference(r);
	}
	else {
	  pips_internal_error("Cast case not implemented");
	}
	break;
      }
    case is_syntax_call :
      {
	/* This may happen when a loop is unstructured by the controlizer */
	c = syntax_call(is);
	l_proper = generic_r_proper_effects_of_call(c);
	break;
      }
    case is_syntax_application :
      {
	  /* This may happen when a structure field contains a pointer to
	     a function. We do not know which function is is... */
	  //application a = syntax_application(is);
	  //expression fe = application_function(a);

	  /* we should try here to retrieve the name of the function through pointer analysis */
	  /* NOT IMPLEMENTED -> returning anywhere read and write, as the function
	     can read and write any global variable as well as it's pointer arguments targets.
           */

	  l_proper = make_anywhere_read_write_memory_effects();

	  /* More effects should be added to take the call site into account */
	  /* Same as for pointer-based call: use type, assume worst case,... */
	  /* A new function is needed to retrieve all functions with a
	     given signature. Then the effects of all the candidates must
	     be unioned. */
	  pips_user_warning("call through a function pointer in a structure -> anywhere effects\n");

	  break;
      }
    case is_syntax_reference:
      {
	// someone typed "i;" in the code... or "a[i++];". It is
	//allowed and may happen in automatic code transformations such as
	//inlining.
	reference r = syntax_reference(is);
	l_proper = generic_proper_effects_of_read_reference(r);
	break;
      }
    case is_syntax_range:
    case is_syntax_sizeofexpression:
    case is_syntax_subscript:
    default :
      pips_internal_error("Instruction expression case %d not implemented",
			  syntax_tag(is));
    }

    pips_debug(2, "Effects for expression instruction in statement%03zd:\n",
	       statement_ordering(current_stat));

    l_proper = gen_nconc(l_proper, effects_dup(cumu_range_effects()));

    if (contract_p)
      l_proper = proper_effects_contract(l_proper);
    ifdebug(2) {
      pips_debug(2, "Proper effects for statement%03zd:\n",
		 statement_ordering(current_stat));
      (*effects_prettyprint_func)(l_proper);
      pips_debug(2, "end\n");
    }

    if(!ENDP(l_proper) && effects_all_read_p(l_proper)) {
      /* The current statement should be ignored as it does not impact
	 the store. Examples in C; "0;" or "i;" or "(void) i". Because
	 PIPS is interprocedural, it may ignore some more statements
	 than gcc. Such statements can be created by program
	 transformations. */
      pips_user_warning("Statement %d is ignored because it does not "
			"modify the store.\n", statement_number(current_stat));
      gen_full_free_list(l_proper);
      l_proper = NIL;
    }
    if (get_constant_paths_p())
      {
	list l_tmp = l_proper;
	l_proper = pointer_effects_to_constant_path_effects(l_proper);
	effects_free(l_tmp);
      }
    store_proper_rw_effects_list(current_stat, l_proper);
  }
}

static void
proper_effects_of_unstructured(unstructured u __attribute__((__unused__)))
{
    statement current_stat = effects_private_current_stmt_head();
    store_proper_rw_effects_list(current_stat,NIL);
}

static bool
loop_filter(loop l)
{
    list l_proper = generic_proper_effects_of_range(loop_range(l));
    list l_eff = gen_nconc(l_proper, effects_dup(cumu_range_effects()));
    current_downward_cumulated_range_effects_push(make_effects(l_eff));
    return(true);
}

static void proper_effects_of_loop(loop l)
{
    statement current_stat = effects_private_current_stmt_head();
    list l_proper = NIL;

    entity i = loop_index(l);
    range r = loop_range(l);

    list li = NIL, lb = NIL;

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat));

    free_cumu_range_effects();
    current_downward_cumulated_range_effects_pop();

    /* proper_effects first */

    /* Effects of loop on loop index.
     * loop index is must-written but may-read because the loop might
     * execute no iterations.
     */
    /* FI, RK: the may-read effect on the index variable is masked by
     * the initial unconditional write on it (see standard page 11-7, 11.10.3);
     * if masking is not performed, the read may prevent privatization
     * somewhere else in the module (12 March 1993)
     */
    /* Parallel case
     *
     * as I need the same effects on a parallel loop to remove
     * unused private variable in rice/codegen.c, I put the
     * same code to compute parallel loop proper effects.
     * this may not be correct, but I should be the only one to use
     * such a feature. FC, 23/09/93
     */
    li = generic_proper_effects_of_written_reference(make_reference(i, NIL));

    /* effects of loop bound expressions. */
    lb = generic_proper_effects_of_range(r);

    l_proper = gen_nconc(li, lb);
    l_proper = gen_nconc(l_proper, effects_dup(cumu_range_effects()));

    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    ifdebug(1) pips_assert("l_proper is consistent",
			   effect_list_consistent_p(l_proper));

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);

    ifdebug(1) pips_assert("l_proper is consistent",
			   effect_list_consistent_p(l_proper));

    if (get_constant_paths_p())
      {
	list l_tmp = l_proper;
	l_proper = pointer_effects_to_constant_path_effects(l_proper);
	effects_free(l_tmp);
      }
    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_forloop(forloop l)
{
    statement current_stat = effects_private_current_stmt_head();
    list l_proper = NIL;

    //    entity i = loop_index(l);
    // range r = loop_range(l);

    list li = NIL, lc = NIL, linc = NIL;

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat));

    /* proper_effects first */

    li = generic_proper_effects_of_expression(forloop_initialization(l));

    /* effects of condition expression */
    lc = generic_proper_effects_of_expression(forloop_condition(l));
    /* effects of incrementation expression  */
    /* we do not know if the incrementation expression will be evaluated -> may effects (see ticket 446) */
    linc = generic_proper_effects_of_expression(forloop_increment(l));
    effects_to_may_effects(linc);

    l_proper = gen_nconc(li, lc);
    l_proper = gen_nconc(l_proper, linc);

    // cumulated range effects are added to internal statements of
    // a la fortran do loops to simulate control effects, but it's unclear
    // whether it should be the same with C for loops which cannot be
    // represented as do loops.
    // free_cumu_range_effects();
    //current_downward_cumulated_range_effects_pop();
    //l_cumu_range = cumu_range_effects();
    //l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range)); 

    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);

    if (get_constant_paths_p())
      {
	list l_tmp = l_proper;
	l_proper = pointer_effects_to_constant_path_effects(l_proper);
	effects_free(l_tmp);
      }
    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_while(whileloop w)
{
    statement current_stat = effects_private_current_stmt_head();
    list /* of effect */ l_proper =
	generic_proper_effects_of_expression(whileloop_condition(w));
    if (get_constant_paths_p())
      {
	list l_tmp = l_proper;
	l_proper = pointer_effects_to_constant_path_effects(l_proper);
	effects_free(l_tmp);
      }

    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_test(test t)
{
    list l_proper=NIL;
    statement current_stat = effects_private_current_stmt_head();

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat));

    /* effects of the condition */
    l_proper = generic_proper_effects_of_expression(test_condition(t));
    l_proper = gen_nconc(l_proper, effects_dup(cumu_range_effects()));

    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);
    if (get_constant_paths_p())
      {
	list l_tmp = l_proper;
	l_proper = pointer_effects_to_constant_path_effects(l_proper);
	effects_free(l_tmp);
      }

    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_sequence(sequence block __attribute__((__unused__)))
{
    statement current_stat = effects_private_current_stmt_head();
    store_proper_rw_effects_list(current_stat, NIL);
}

static bool stmt_filter(statement s)
{
  pips_debug(1, "Entering statement with ordering: %03zd and number: %03zd\n",
	     statement_ordering(s), statement_number(s));
  effects_private_current_stmt_push(s);
  effects_private_current_context_push((*load_context_func)(s));
  return(true);
}

/**
 @param entity is a
 @param
 @return : a list of effects corresponding to the effects of the rhs if there
           is an initialization in the declaration., plus the effects of the lhs
*/
static list generic_proper_effects_of_declaration(entity decl)
{
  list l_eff = NIL;
  pips_debug(1, "declaration of entity %s \n", entity_local_name(decl));

  if(type_variable_p(entity_type(decl)))
    {
      value v_init = entity_initial(decl);

      pips_debug(1, "begin\n");
      /* generate effects due to the initialisation */
      if (value_expression_p(v_init))
	{
	  expression exp_init = value_expression(v_init);
	  l_eff = generic_proper_effects_of_expression(exp_init);
	}

      /* if there is an initial value, and if the variable is not a static one,
	 then there is a write on the entity (well on  the reference constituted
	 by the entity name with no indices !).
	 There may be a memory leak here because we do not want a preference in
	 the effect. However, I do not have a good solution for the time being
	 because in declarations, the left hand side is not a reference. BC.
	 I should may be call generic_proper_effects_of lhs instead, but the case is
	 slightly different for arrays. Or directly (*reference_to_effect_func) in case of a scalar ?
      */
      /* To avoid problems with loop distribution, we need a (write)
	 effect whether there is an initialization or not. This is
	 consistent with transformers and preconditions, but not with
	 the used before set analysis. In the short term, you can
	 comment out the second part of the condition to solve the
	 loop distribution issue. */
      if (!variable_static_p(decl) && !value_unknown_p(v_init))
	{
	  type decl_t = entity_basic_concrete_type(decl);
	  list l_tmp = NIL;

	  if (!ENDP(variable_dimensions(type_variable(decl_t))))
	    {
	      effect decl_eff = (*reference_to_effect_func)(make_reference(decl,NIL), make_action_write_memory(), true);
	      l_tmp =  generic_effect_generate_all_accessible_paths_effects(decl_eff, decl_t, is_action_write);
	    }
	  else
	    {
	      // make sure the make_reference does not create a memory leak
	      /* The other problem is that it generates a "is referenced" effect*/
	      l_tmp = generic_proper_effects_of_reference(make_reference(decl,
									 NIL),
							  true);
	    }
	  l_eff= gen_nconc(l_eff, l_tmp);
	}
      if (contract_p)
	l_eff = proper_effects_contract(l_eff);
      pips_debug_effects(1, "ending with:", l_eff);
    }
  return l_eff;
}

effect make_declaration_effect(entity e, bool written_p)
{
  effect eff = effect_undefined;
  /* FI: generate a declaration or a type write */

  reference r = make_reference(e, NIL);
  action a = action_undefined;
  action_kind ak = action_kind_undefined;
  /* FI: I'm not sure this is a very generic decision; I do not
     foresee how predicates could refine dependences. But we already
     missed that for scalar variables:-( */
  descriptor d = make_descriptor_none();
  cell c = make_cell_reference(r);
  approximation ap = make_approximation_exact();

  if(typedef_entity_p(e)) {
    /* FI: more work here; you have to check that other typedefed
       types or variables are not used in this typedef. Quite a
       recursive search... */
    ak = make_action_kind_type_declaration();
  }
  else
    ak = make_action_kind_environment();

  if(written_p)
    a = make_action_write(ak);
  else
    a = make_action_read(ak);

  eff = make_effect(c, a, ap, d);

  return eff;
}

static void proper_effects_of_statement(statement s)
{
  /* Handling of declarations attached to a declaration statement

  */
  pips_debug(1, "statement%03zd :\n", statement_number(s));
  if (c_module_p(get_current_module_entity()) &&
      (declaration_statement_p(s) /*|| block_statement_p(s)*/ ))
    {
      list l_eff = NIL;
      list l_decls = statement_declarations(s);

      pips_debug(1, "declaration statement \n");

      FOREACH(ENTITY, e, l_decls)
	{
	  if(!get_bool_property("MEMORY_EFFECTS_ONLY")) {
	    effect de = make_declaration_effect(e, true);
	    type vt = entity_type(e);

	    l_eff = gen_nconc(l_eff, CONS(EFFECT, de, NIL));
	    // Should be put first or last?
	    //l_eff = CONS(EFFECT, re, l_eff);

	    if(typedef_type_p(vt)) {
	      entity te = basic_typedef(variable_basic(type_variable(vt)));
	      effect tre = make_declaration_effect(te, false); // type
	      // reference effect
	      l_eff = gen_nconc(l_eff, CONS(EFFECT, tre, NIL));
	      //l_eff = CONS(EFFECT, tre, l_eff);
	    }
	  }
	  l_eff = gen_nconc(l_eff,generic_proper_effects_of_declaration(e));
	}
      if (get_constant_paths_p())
	{
	  list l_tmp = l_eff;
	  l_eff = pointer_effects_to_constant_path_effects(l_eff);
	  effects_free(l_tmp);
	}

      if(bound_proper_rw_effects_p(s))
	{
	  l_eff = gen_nconc(l_eff,
			    load_proper_rw_effects_list(s));
	  update_proper_rw_effects_list(s, l_eff);
	}
      else
	store_proper_rw_effects_list(s,l_eff);
    }

  if (!bound_proper_rw_effects_p(s))
    {
      pips_debug(2, "Warning, proper effects undefined, set to NIL\n");
      store_proper_rw_effects_list(s,NIL);
    }
  effects_private_current_stmt_pop();
  effects_private_current_context_pop();

  pips_debug(1, "End statement%03zd :\n", statement_number(s));

}

void proper_effects_of_module_statement(statement module_stat)
{
    make_effects_private_current_stmt_stack();
    make_effects_private_current_context_stack();
    make_current_downward_cumulated_range_effects_stack();
    pips_debug(1,"begin\n");

    gen_multi_recurse
	(module_stat,
	 statement_domain, stmt_filter, proper_effects_of_statement,
	 sequence_domain, gen_true, proper_effects_of_sequence,
	 test_domain, gen_true, proper_effects_of_test,
	 /* Reached only through syntax (see expression rule) */
	 call_domain, gen_true, proper_effects_of_call,
	 loop_domain, loop_filter, proper_effects_of_loop,
	 whileloop_domain, gen_true, proper_effects_of_while,
	 forloop_domain, gen_true, proper_effects_of_forloop,
	 unstructured_domain, gen_true, proper_effects_of_unstructured,
         /* Just to retrieve effects of instructions with kind
	    expression since they are ruled out by the next clause */
	 instruction_domain, gen_true, proper_effects_of_expression_instruction,
	 expression_domain, gen_false, gen_null, /* NOT THESE CALLS */
	 NULL);

    pips_debug(1,"end\n");
    free_effects_private_current_stmt_stack();
    free_effects_private_current_context_stack();
    free_current_downward_cumulated_range_effects_stack();
}

bool proper_effects_engine(const char *module_name)
{
    /* Get the code of the module. */
    set_current_module_statement( (statement)
		      db_get_memory_resource(DBR_CODE, module_name, true));

    set_current_module_entity(module_name_to_entity(module_name));

    (*effects_computation_init_func)(module_name);

    /* Compute the effects or references of the module. */
    init_proper_rw_effects();

    if (get_pointer_info_kind() == with_points_to)
      set_pt_to_list( (statement_points_to)
			   db_get_memory_resource(DBR_POINTS_TO, module_name, true) );
    else if (get_pointer_info_kind() == with_pointer_values)
      set_pv( db_get_simple_pv(module_name));

    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    proper_effects_of_module_statement(get_current_module_statement());

    pips_debug(1, "end\n");
    debug_off();

    (*db_put_proper_rw_effects_func)(module_name, get_proper_rw_effects());

     if (get_pointer_info_kind() == with_points_to)
       reset_pt_to_list();
     else if (get_pointer_info_kind() == with_pointer_values)
       reset_pv();

    reset_current_module_entity();
    reset_current_module_statement();
    reset_proper_rw_effects();

    (*effects_computation_reset_func)(module_name);

    return(true);
}




/* compute proper effects for both expressions and statements
   WARNING: the functions are set as a side effect.
 */
void
expression_proper_effects_engine(
    const char* module_name,
    statement current)
{
    (*effects_computation_init_func)(module_name);

    init_proper_rw_effects();
    init_expr_prw_effects();

    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    proper_effects_of_module_statement(current);

    pips_debug(1, "end\n");
    debug_off();

    (*effects_computation_reset_func)(module_name);
}
