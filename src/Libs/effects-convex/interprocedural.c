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
/* package regions :  Alexis Platonoff, 22 Aout 1990, Be'atrice Creusillet 10/94
 *
 * interprocedural
 * ---------------
 *
 * This File contains the main functions that compute the interprocedural
 * translation of regions (forward and backward).
 *
 * Vocabulary : _ A variable refered as a "region" is in fact of the NEWGEN
 *                type "effect". The use of the word "region" allows to keep
 *                the difference with the effects package.
 *              _ The word "func" always refers to the external called 
 *                subroutine.
 *              _ The word "real" always refers to the calling subroutine 
 */

#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "misc.h"
#include "semantics.h"
#include "text.h"
#include "text-util.h"
#include "preprocessor.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "sc.h"
#include "polyedre.h"

#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"

#define IS_EG TRUE
#define NOT_EG FALSE

#define PHI_FIRST TRUE
#define NOT_PHI_FIRST FALSE

#define BACKWARD TRUE
#define FORWARD FALSE

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))



/* jmp_buf overflow_error;*/


/*********************************************************** INITIALIZATION */

void convex_regions_translation_init(entity callee, list real_args, bool backward_p )
{

  set_interprocedural_translation_context_sc(callee, real_args);
  if (backward_p)
    set_backward_arguments_to_eliminate(callee);
  else
    set_forward_arguments_to_eliminate();
}

void convex_regions_translation_end()
{
    reset_translation_context_sc();
    reset_arguments_to_eliminate();
}


/************************************************************** INTERFACES  */

static statement current_stmt = statement_undefined;
static entity current_callee = entity_undefined;
static list l_sum_out_reg = list_undefined;

void reset_out_summary_regions_list()
{
    l_sum_out_reg = list_undefined;
}

void update_out_summary_regions_list(list l_out)
{
    if (list_undefined_p(l_sum_out_reg))
	l_sum_out_reg = l_out;
    else
	l_sum_out_reg = RegionsMayUnion(l_sum_out_reg, l_out,
					regions_same_action_p);
}

list get_out_summary_regions_list()
{
    return(l_sum_out_reg);
}

static bool stmt_filter(s)
statement s;
{
    pips_debug(1, "statement %td\n", statement_number(s));
    
    current_stmt = s;
    return(TRUE);
}


list out_regions_from_caller_to_callee(entity caller, entity callee)
{
    char *caller_name;
    statement caller_statement;

    reset_current_module_entity();
    set_current_module_entity(caller);
    caller_name = module_local_name(caller);
    pips_debug(2, "begin for caller: %s\n", caller_name);
    
    /* All we need to perform the translation */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, caller_name, TRUE) );
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, caller_name, TRUE));
    module_to_value_mappings(caller);
    set_precondition_map( (statement_mapping) 
        db_get_memory_resource(DBR_PRECONDITIONS, caller_name, TRUE));

    set_out_effects( (statement_effects) 
	db_get_memory_resource(DBR_OUT_REGIONS, caller_name, TRUE) );

    caller_statement = (statement) 
	db_get_memory_resource (DBR_CODE, caller_name, TRUE);

    current_callee = callee;
    gen_multi_recurse(caller_statement,
		      statement_domain, stmt_filter, gen_null,
		      call_domain, out_regions_from_call_site_to_callee, gen_null,
		      NULL);
  
    reset_current_module_entity();
    set_current_module_entity(callee);

    free_value_mappings();

    reset_cumulated_rw_effects();
    reset_precondition_map();
    reset_out_effects();
    reset_current_module_statement();
    free_value_mappings();
    pips_debug(2, "end\n");
    return(l_sum_out_reg);
}

/* void out_regions_from_call_site_to_callee(call c)
 * input    : a potential call site for current_callee.
 * output   : nothing
 * modifies : l_sum_out_reg becomes the may union of l_sum_out_reg and 
 *            the translated out regions of the current call site. 
 * comment  :
 */
void out_regions_from_call_site_to_callee(call c)
{
    transformer context;
    list l_out = NIL, l_tmp = NIL;

    if (call_function(c) != current_callee)
	return;

    context= load_statement_precondition(current_stmt);
    l_out = load_statement_out_regions(current_stmt);
    
    l_tmp = regions_forward_translation(current_callee, call_arguments(c), l_out,
					context);
    update_out_summary_regions_list(l_tmp);
}


/* list in_regions_of_external(entity func, list real_args, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list in_regions_of_external(func, real_args, context)
entity func;
list real_args;
transformer context;
{
    list le = NIL;
    char *func_name = module_local_name(func);

    pips_debug(4, "translation regions for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("in_region_of_external", "%s: bad function\n", func_name);
    }
    else 
    {
	list func_regions;

        /* Get the regions of "func". */
	func_regions = effects_to_list((effects)
	    db_get_memory_resource(DBR_IN_SUMMARY_REGIONS, func_name, TRUE));
	/* translate them */
	le = regions_backward_translation(func, real_args, func_regions, context,
					  SUMMARY);
    }
    return le;
}


/* list regions_of_external(entity func, list real_args, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list regions_of_external(entity func,list real_args,transformer context,
			 bool proper)
{
    list le = NIL;
    char *func_name = module_local_name(func);

    pips_debug(4, "translation regions for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("region_of_external", "%s: bad function\n", func_name);
    }
    else 
    {
	list func_regions;

        /* Get the regions of "func". */
	func_regions = effects_to_list((effects)
	    db_get_memory_resource(DBR_SUMMARY_REGIONS, func_name, TRUE));
	/* translate them */
	le = regions_backward_translation(func, real_args, func_regions, context,
					  proper);
    }
    return le;
}

list /* of effects */
convex_regions_backward_translation(entity func, list real_args,
				    list l_reg, transformer context)
{
    list l_res = NIL;
    
    l_res = regions_backward_translation(func, real_args, l_reg, context, TRUE);
    
    return l_res;
}

list /* of effects */
convex_regions_forward_translation(entity callee, list real_args,
				    list l_reg, transformer context)
{
    list l_res = NIL;
    
    if(fortran_module_p(callee) && fortran_module_p(get_current_module_entity()))
      l_res = regions_forward_translation(callee, real_args, l_reg, context);
    else if (c_module_p(callee) && c_module_p(get_current_module_entity()))
      l_res = generic_c_effects_forward_translation(callee, real_args, l_reg, context);
    return l_res;
}


/***************************************************** BACKWARD TRANSLATION */

static list formal_regions_backward_translation(entity func, list real_args, 
						list func_regions, 
						transformer context);
static list common_regions_backward_translation(entity func, list func_regions);
static list common_region_translation(entity func, region reg, boolean backward);

/* list regions_backward_tranlation(entity func, list real_args, 
 *                                  list func_regions, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list regions_backward_translation(entity func, list real_args,
				  list func_regions,
				  transformer context, bool proper)
{
    list le = NIL;
    list tce, tfe;

    ifdebug(4)
    {
	pips_debug(4,"Initial regions\n");
	print_regions(func_regions);
    }

    set_interprocedural_translation_context_sc(func,real_args);
    set_backward_arguments_to_eliminate(func);

    /* Compute the regions on formal variables. */
    tfe = formal_regions_backward_translation(func,real_args,func_regions,context);
    
    /* Compute the regions on common variables (static & global variables). */
    tce = common_regions_backward_translation(func, func_regions);
    
    if (proper)
	le = gen_nconc(tce,tfe);
    else
	le = RegionsMustUnion(tce, tfe, regions_same_action_p);		
    
    /* FI: add local precondition (7 December 1992) */
    le = regions_add_context(le, context);
    
    ifdebug(4)
    {
	pips_debug(4, " Translated_regions :\n");
	print_regions(le);
    }
    
    reset_translation_context_sc();
    reset_arguments_to_eliminate();
    return(le);
}


/* static list formal_regions_backward_translation(entity func, list real_args,
 *                                   func_regions, transformer context)
 * input    : an external function func, its real arguments at call site 
 *            (real_args),
 *            its summary regions (with formal args), and the calling context. 
 * output   : the translated formal regions.
 * modifies : ?
 * comment  :	
 *
 * Algorithm :
 * -----------
 *    let func_regions be the list of the regions on variables of func
 *    let real_regions be the list of the translated regions on common variables
 *
 *    real_regions = empty
 *    FOR each expression real_exp IN real_args
 *        arg_num = number in the list of the function real arguments
 *        FOR each func_reg IN func_regions
 *            func_ent = entity of the region func_reg
 *            IF func_ent is the formal parameter numbered arg_num
 *                IF real_exp is an lhs (expression with one entity)
 *                    real_reg = translation of the region func_reg
 *                    real_regions = (real_regions) U {real_reg}
 *                ELSE
 *                    real_regions = (real_regions) U
 *                                   (regions of the expression real_exp)
 *                ENDIF
 *            ENDIF
 *        ENDFOR
 *    ENDFOR
 */
static list formal_regions_backward_translation(func, real_args, func_regions, 
						context)
entity func;
list real_args, func_regions;
transformer context;
{
    list real_regions = NIL, r_args;
    int arg_num;

    pips_debug(8, "begin\n");

    for (r_args = real_args, arg_num = 1; r_args != NIL;
	 r_args = CDR(r_args), arg_num++) 
    {
	MAP(EFFECT, func_reg,
	 {
	     entity func_ent = region_entity(func_reg);
	     
	     /* If the formal parameter corresponds to the real argument then
	      * we perform the translation.
	      */
	     if (ith_parameter_p(func, func_ent, arg_num))
	     {
		 expression real_exp = EXPRESSION(CAR(r_args));
		 syntax real_syn = expression_syntax(real_exp);
		 
		 /* If the real argument is a reference to an entity, then we
		  * translate the regions of the corresponding formal parameter
		  */
		 if (syntax_reference_p(real_syn)) 
		 {
		    reference real_ref = syntax_reference(real_syn);
		    list real_inds = reference_indices(real_ref);
		    entity real_ent = reference_variable(real_ref);
		    region real_reg;
		    real_reg =
			region_translation(func_reg, func, reference_undefined,
				  real_ent, get_current_module_entity(), real_ref,
				  VALUE_ZERO, BACKWARD);
		    
		    real_regions = regions_add_region(real_regions, real_reg);
		    /* The indices of the reference are always evaluated */
		    if (! ENDP(real_inds)) 
			real_regions = gen_nconc
			    (real_regions,
			     proper_regions_of_expressions(real_inds, context));
		}
		 /* Else, the real argument is a complex expression, which
		  * is merely evaluated during execution of the program; 
		  * Since Fortran forbids write effects on expressions
		  * passed as arguments, the regions on the formal parameter 
		  * are merely ignored. The regions computed are those of the
		  * real parameter expression. 
		  */
		 else 
		 {
		    real_regions =
			gen_nconc(real_regions,
				  generic_proper_effects_of_expression(real_exp)); 
		}
	     }
	 }, func_regions);
    }

    ifdebug(5) 
    {
       pips_debug(5, "proper real regions\n");
       print_regions(real_regions);
    }
    return(real_regions);
}

static list common_regions_backward_translation(entity func, list func_regions) 
{
    list real_regions = NIL;

    MAP(EFFECT, func_reg, 
    {
	/* we are only interested in regions concerning common variables. 
	 * They are  the entities with a ram storage. They can not be dynamic
         * variables, because these latter were eliminated of the code_regions
         * (cf. region_of_module). */
	if (storage_ram_p(entity_storage(region_entity(func_reg))))
	{
	    list regs = common_region_translation(func, func_reg, BACKWARD);
	    real_regions = RegionsMustUnion(real_regions, regs, 
					    regions_same_action_p);	
	}
    },
	func_regions);

    return(real_regions);

}



/**

 @param l_sum_eff is a list of effects on a C function formal parameter. These
        effects must be vissible from the caller, which means that their 
        reference has at leat one index.       
 @param real_arg is an expression. It's the real argument corresponding to
        the formal parameter which memory effects are represented by l_sum_eff.
 @param context is the transformer translating the callee's neame space into 
        the caller's name space.	 
 @return a list of effects which are the translation of l_sum_eff in the
         caller's name space.
 */
list c_convex_effects_on_formal_parameter_backward_translation(list l_sum_eff, 
						  expression real_arg, 
						  transformer context)
{
  list l_eff = NIL; /* the result */
  syntax real_s = expression_syntax(real_arg);
  type real_arg_t = expression_to_type(real_arg);
  
  
  ifdebug(5)
    {
      pips_debug(8, "begin for real arg %s, of type %s and effects :\n", 
		 words_to_string(words_expression(real_arg)),
		 type_to_string(real_arg_t));
      (*effects_prettyprint_func)(l_sum_eff);		 
    }
  
  switch (syntax_tag(real_s))
    {
    case is_syntax_reference:
      {  
	reference real_ref = syntax_reference(real_s);
	entity real_ent = reference_variable(real_ref);
	list real_ind = reference_indices(real_ref);
	
	/* if it's a pointer or a partially indexed array 
	 * We should do more testing here to check if types
	 * are compatible... (see effect_array_substitution ?)
	 */

	/* the test here may not be right. I guess I should use basic_concrete_type here BC */
	if (pointer_type_p(real_arg_t) ||
	    gen_length(real_ind) < type_depth(entity_type(real_ent)))
	  {
	    
	    FOREACH(EFFECT, eff, l_sum_eff)
	      {
		
		reference new_ref = copy_reference(real_ref);
		effect new_eff = effect_undefined;
		
		pips_debug(8, "pointer type real arg reference\n");
		
		
		/* Then we compute the region corresponding to the
		   real argument
		*/
		pips_debug(8, "effect on the pointed area : \n");
		new_eff = (*reference_to_effect_func)
		  (new_ref, effect_action_tag(eff), false);		

		/* this could easily be made generic BC. */
		if(!anywhere_effect_p(new_eff))
		  {
		    effect init_eff = (*effect_dup_func)(eff);
		    /* we translate the initial region descriptor
		       into the caller's name space
		    */
		    convex_region_descriptor_translation(init_eff);		
		    /* and we "append" the initial region to the real arg
		       region.
		    */
		    new_eff = region_append(new_eff, init_eff);	
		    /* shouldn't it be a union ? BC */
		    l_eff = gen_nconc(l_eff, CONS(EFFECT, new_eff, NIL));
		    free_effect(init_eff);
		  }
		
	      }
	    
	  } /*  if (pointer_type_p(real_arg_t)) */
	else
	  {
	    pips_debug(8, "real arg reference is not a pointer and is not a partially indexed array -> NIL \n");
	    
	  } /* else */
	break;
      } /* case is_syntax_reference */ 
    case is_syntax_subscript:
      {
	pips_internal_error("Subscript not supported yet\n");
	break;
      }
    case is_syntax_call:
      {
	call real_call = syntax_call(real_s);
	entity real_op = call_function(real_call);
	list args = call_arguments(real_call);
	effect n_eff = effect_undefined;
	
	if (ENTITY_ASSIGN_P(real_op))
	  {
	    l_eff = c_convex_effects_on_formal_parameter_backward_translation
	      (l_sum_eff, EXPRESSION(CAR(CDR(args))), context);
	  }
	else if(ENTITY_ADDRESS_OF_P(real_op)) 
	  {
	    expression arg1 = EXPRESSION(CAR(args));
	    list l_real_arg = NIL;
	    effect eff_real;
	    
	    /* first we compute an effect on the argument of the 
	       address_of operator (to treat cases like &(n->m))*/
	    pips_debug(6, "addressing operator case \n");

	    l_real_arg = 
	      generic_proper_effects_of_complex_address_expression
	      (arg1, &eff_real, true);

	    ifdebug(6)
		{
		  pips_debug(6, "base effect :\n");
		  print_region(eff_real);
		}
	      
	    FOREACH(EFFECT, eff, l_sum_eff)
	      {
		reference eff_ref = effect_any_reference(eff);
		list eff_ind = reference_indices(eff_ref);
		int nb_phi_eff = (int) gen_length(eff_ind);
		int nb_phi_n_eff;

		ifdebug(6)
		  {
		    pips_debug(6, "current formal effect :\n");
		    print_region(eff);
		  }
		
		if (effect_undefined_p(eff_real) || anywhere_effect_p(eff_real))
		  {
		    n_eff =  make_anywhere_effect(effect_action_tag(eff));
		    l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));
		  }
		else {
		  n_eff = (*effect_dup_func)(eff_real);
		  effect_approximation_tag(n_eff) = 
		    effect_approximation_tag(eff);
		  effect_action_tag(n_eff) = effect_action_tag(eff);
		  nb_phi_n_eff = (int) 
		    gen_length(reference_indices(effect_any_reference(n_eff)));

		  if(!ENDP(eff_ind))
		    {
		      effect eff_init = (*effect_dup_func)(eff);
		      Psysteme sc_init ;
		      Psysteme sc_n_eff = region_system(n_eff);
		      int i;
		      
		      /* we translate the initial region descriptor
			 into the caller's name space
		      */
		      convex_region_descriptor_translation(eff_init);
		      sc_init = region_system(eff_init);
			  
			  
		      if (nb_phi_n_eff !=0)
			{
			  
			  /* the first index of eff_init is added to the last index
			     of n_eff, and the other indexes of eff_init are 
			     appended to n_eff
			  */
			  
			  /* preparing the system of eff_init */
			  /* first rename phi1 into psi1 */
			  entity phi1 = make_phi_entity(1);
			  entity psi1 = make_psi_entity(1);
			  sc_init = sc_variable_rename(sc_init, (Variable) phi1, (Variable) psi1);
			  
			  /* then translate other phi variables */
			  for(i=nb_phi_eff; i>1; i--)
			    {
			      entity old_phi = make_phi_entity(i);
			      entity new_phi = make_phi_entity(nb_phi_n_eff+i-1);
			      
			      sc_init = sc_variable_rename(sc_init, (Variable) old_phi, (Variable) new_phi);	  
			    }
			  
			  /* preparing the system of n_eff */
			  entity phi_max_n_eff = make_phi_entity(nb_phi_n_eff);
			  entity rho_max_n_eff = make_rho_entity(nb_phi_n_eff);
			  
			  sc_n_eff = sc_variable_rename(sc_n_eff, (Variable) phi_max_n_eff, (Variable) rho_max_n_eff);
			  region_system(n_eff) = sc_n_eff;

			  pips_debug_effect(8, "n_eff after variable renaming: \n", n_eff);
			  
			  /* then we append sc_init to sc_n_eff
			   */
			  region_sc_append_and_normalize(n_eff, sc_init, TRUE);
			  
			  pips_debug_effect(8, "n_eff after appending sc_init: \n", n_eff);
			  
			  /* then we add the constraint phi_max_n_eff = psi1 + rho_max_n_eff 
			     and we eliminate psi1 and rho_max_n_eff
			  */
			  Pvecteur v_phi_max_n_eff = vect_new((Variable) phi_max_n_eff, VALUE_ONE);
			  Pvecteur v_psi1 = vect_new((Variable) psi1, VALUE_ONE);
			  Pvecteur v_rho_max_n_eff = vect_new((Variable) rho_max_n_eff, VALUE_ONE);
			  sc_n_eff = region_system(n_eff);
			  v_phi_max_n_eff = vect_substract(v_phi_max_n_eff, v_psi1);
			  v_phi_max_n_eff = vect_substract(v_phi_max_n_eff, v_rho_max_n_eff);
			  sc_constraint_add(sc_n_eff, contrainte_make(v_phi_max_n_eff), TRUE);
			  region_system(n_eff) = sc_n_eff;
			  			  
			  region_remove_psi_variables(n_eff);
			  region_remove_rho_variables(n_eff);
			  
			} /*  if (nb_phi_n_eff !=0) */
		      else
			{
			  /* if it's a scalar, but not a pointer, n_eff is OK */
			  /* if it's a pointer, n_eff is equal to eff but for the first
			     dimension, which should be equal to 0 in eff (I do not check
			     that here, because it is checked before in simple effects).
			  */
			  entity n_eff_ent = reference_variable(effect_any_reference(n_eff));
			  type bct = basic_concrete_type(entity_type(n_eff_ent));

			  if (derived_type_p(bct) || pointer_type_p(bct))
			    {
			      reference ref;
			      Psysteme sc;
			      sc_rm(sc_n_eff);
			      region_system(n_eff) = sc_dup(sc_init);

			      /* first remove the phi1 variable */
			      entity phi1 = make_phi_entity(1);
			      list l_tmp = CONS(ENTITY, phi1, NIL);
			      region_exact_projection_along_variables(n_eff, l_tmp);
			      gen_free_list(l_tmp);
			      sc = region_system(n_eff);

			      /* then rename all the phi variables in reverse order */
			      for(i=2; i<=nb_phi_eff; i++)
				{
				  entity old_phi = make_phi_entity(i);
				  entity new_phi = make_phi_entity(i-1);
				  
				  sc_variable_rename(sc, old_phi, new_phi);	  
				}

			      if (cell_preference_p(effect_cell(n_eff)))
				{
				  /* it's a preference : we should not modify it */
				  pips_debug(8, "It's a preference\n");
				  ref = copy_reference(preference_reference(cell_preference(effect_cell(n_eff))));
				  preference_reference(cell_preference(effect_cell(n_eff))) = ref;
				}
			      else
				{
				  /* it's a reference : let'us modify it */
				  ref = cell_reference(effect_cell(n_eff));
				}
			      int i;
			      for(i = 1; i<nb_phi_eff; i++)
				{
				  reference_indices(ref) = gen_nconc(reference_indices(ref), 
								     CONS(EXPRESSION, copy_expression(EXPRESSION(CAR(eff_ind))), NIL));
				  POP(eff_ind);
								     
				}
			    }
			  free_type(bct);
			}
		      		      		     		  
		    } /* if(!ENDP(eff_ind))*/
		  
		} /* else du if (effect_undefined_p(eff_real) || ...) */
						
		l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));
	      } /*  FOREACH(EFFECT, eff, l_sum_eff) */

	    gen_free_list(l_real_arg);
	    free_effect(eff_real);
	    
	  }
	else if(ENTITY_POINT_TO_P(real_op)|| ENTITY_FIELD_P(real_op))
	  {
	    list l_real_arg = NIL;
	    effect eff_real;
	    /* first we compute an effect on the real_arg */
	    
	    l_real_arg = generic_proper_effects_of_complex_address_expression
	      (real_arg, &eff_real, true);

	     FOREACH(EFFECT, eff, l_sum_eff)
	      {
		effect eff_formal = (*effect_dup_func)(eff);
		effect new_eff;

		if (effect_undefined_p(eff_real))
		  new_eff =  make_anywhere_effect(effect_action_tag(eff));
		else
		  {
		    new_eff = (*effect_dup_func)(eff_real);
		    effect_approximation_tag(new_eff) = 
		      effect_approximation_tag(eff);
		    effect_action_tag(new_eff) =
		      effect_action_tag(eff);		    
		  		    		
		  
		    /* first we translate the formal region predicate */
		    convex_region_descriptor_translation(eff_formal);

		    /* Then we append the formal region to the real region */
		    /* Well this is valid only in the general case :
		     * we should verify that types are compatible. */
		    new_eff = region_append(new_eff, eff_formal);	
		    /* shouldn't it be a union ? BC */
		    l_eff = gen_nconc(l_eff, CONS(EFFECT, new_eff, NIL));
		    free_effect(eff_formal);
		    
		  } /* else du if (effect_undefined_p(eff_real)) */
		l_eff = gen_nconc(l_eff, CONS(EFFECT, new_eff, NIL));
	      } /* FOREACH(EFFECT, eff, l_sum_eff) */
	     gen_free_list(l_real_arg);
	     free_effect(eff_real);
	     
	  }
	else if(ENTITY_MALLOC_SYSTEM_P(real_op)) 
	  {
	    /* BC : do not generate effects on HEAP */
	    /*n_eff = heap_effect(get_current_module_entity(),
	      copy_action(effect_action(eff)));*/
	  }
	else 
	  {
	    l_eff = gen_nconc
	      (l_eff, 
	       c_actual_argument_to_may_summary_effects(real_arg, 'x'));
	  }
	
	if (n_eff != effect_undefined && l_eff == NIL)
	  l_eff = CONS(EFFECT,n_eff, NIL);
	break;
      } /* case is_syntax_call */
    case is_syntax_cast :
      {
	/* Ignore the cast */
	cast c = syntax_cast(real_s);
	pips_user_warning("Cast effect is ignored\n");
	l_eff = c_convex_effects_on_formal_parameter_backward_translation
	  (l_sum_eff, cast_expression(c), context);
	break;
      }
    case is_syntax_sizeofexpression :
      {
	pips_debug(5,"sizeof expression -> NIL");
	break;
      }
    case is_syntax_va_arg :
      {
	pips_internal_error("va_arg() : should have been treated before\n");
	break;
      }
    case is_syntax_application :
      {
	pips_internal_error("Application not supported yet\n");
	break;
      }
    case is_syntax_range :
      {
	pips_user_error("Illegal effective parameter: range\n");
	break;
      }
    default:
      pips_internal_error("Illegal kind of syntax\n");
    } /* switch */
  
  free_type(real_arg_t);
  

  ifdebug(8)
    {
      pips_debug(8, "end with effects :\n");
      print_regions(l_eff);		 
    }
  
  return(l_eff); 
}




/****************************************************** FORWARD TRANSLATION */


static list real_regions_forward_translation(entity func, list real_args, 
					     list l_reg, transformer context);
static list common_regions_forward_translation(entity func, list real_regions);


/* list regions_forward_translation(entity func, list real_args, l_reg, 
 *                                  transformer context
 * input    : the called function func, the real arguments of the call,
 *            the list of regions to translate, and the context of the call.
 * output   : the translated list of regions : real arguments are translated 
 *            into formal arguments, and common variables of the caller into 
 *            common variables of the callee. 
 * modifies : nothing.
 * comment  :	
 */
list regions_forward_translation(func, real_args, l_reg, context)
entity func;
list real_args, l_reg;
transformer context;
{
    list l_t_reg = NIL;
    list l_form_reg, l_common_reg;

    ifdebug(3)
    {
	pips_debug(3,"initial regions :\n");
	print_regions(l_reg);
    }

    set_interprocedural_translation_context_sc(func,real_args);
    set_forward_arguments_to_eliminate();

    l_form_reg = real_regions_forward_translation
	(func, real_args, l_reg, context);
    l_common_reg = common_regions_forward_translation(func, l_reg);
    l_t_reg = RegionsMustUnion
	(l_form_reg, l_common_reg, regions_same_action_p);

    ifdebug(3)
    {
	pips_debug(3,"final regions : \n");
	print_regions(l_t_reg);
    }

    reset_translation_context_sc();
    reset_arguments_to_eliminate();
    return l_t_reg;
}


/* static list real_regions_forward_translation(entity func, list real_args, l_reg, 
 *                                              transformer context)
 * input    : the called function func, the real arguments of the call,
 *            the list of regions to translate, and the context of the call.
 * output   : the list of translated regions correponding to the formal arguments
 *            of the called function.
 * modifies : l_reg and the regions it contains.
 * comment  : 
 *            for each real argument in real_args
 *              if it is a reference
 *                for each region in l_reg
 *                  if the current region concerns the current real argument
 *                    if the corresponding formal parameter is a scalar
 *                      the translated region is a scalar region, which
 *                      reference is the formal argument, and which 
 *                      action and approximation are those of the initial
 *                      region.
 *                    else it is an array, 
 *                      and the tranlation is performed by
 *                      another procedure.
 *                endfor
 *              else, it is a complex expression
 *                we search the regions in l_reg corresponding to
 *                the elements accessed in the complex expression.
 *                and we make a read region corresponding to the 
 *                formal scalar parameter.
 *              endif
 *            endfor
 *               
 */
static list real_regions_forward_translation(func, real_args, l_reg, context)
entity func;
list real_args, l_reg;
transformer context;
{
    entity caller = get_current_module_entity();
    int arg_num;
    list l_formal = NIL;
    list r_args = real_args;
    list l_sum_rw_reg = 
	effects_to_list((effects) db_get_memory_resource
			(DBR_SUMMARY_REGIONS,
			 module_local_name(func),
			 TRUE));
    
    /* for each actual parameter expression, we search in the actual regions 
     * the corresponding elements. If it exists, we make the corresponding 
     * regions, and translate them */

    ifdebug(8) 
    {
	pips_debug(8,"initial regions :\n");
	print_regions(l_reg);
    }

    for (arg_num = 1; !ENDP(r_args); r_args = CDR(r_args), arg_num++) 
    {
	expression real_exp = EXPRESSION(CAR(r_args));
	entity formal_ent = find_ith_formal_parameter(func, arg_num);

	if (syntax_reference_p(expression_syntax(real_exp)))
	{
	    reference real_ref = syntax_reference(expression_syntax(real_exp));
	    entity real_ent = reference_variable(real_ref);

	    MAP(EFFECT, reg,
	     {
		 entity reg_ent = region_entity(reg);

		 pips_debug(8, " real = %s, formal = %s \n", 
			    entity_name(real_ent), entity_name(reg_ent));
		 
		 if (same_entity_p(reg_ent , real_ent)) 
		 {
		     region formal_reg;
		     formal_reg = region_translation(
			 reg, caller, real_ref,
			 formal_ent, func, reference_undefined, 
			 VALUE_ZERO, FORWARD);		     
		     l_formal = RegionsMustUnion(
			 l_formal, 
			 CONS(EFFECT, formal_reg, NIL), 
			 regions_same_action_p);
		 }
	     }, l_reg);
	    
	} /* if */
	else 
	{
	    /* REVOIR ICI */
	    list l_exp_reg = regions_of_expression(real_exp, context);
	    list l_real_exp = 
		RegionsIntersection(l_exp_reg, effects_dup(l_reg),
				    regions_same_action_p);

	    pips_debug(8, "real argument is a complex expression \n" 
		"\tit can not correspond to a written formal parameter.\n");
	    
	    if (!ENDP(l_real_exp)) 
	    {
		region  formal_reg = 
		    reference_whole_region(make_regions_reference(formal_ent),
				     is_action_read);
		effect_to_may_effect(formal_reg);
		l_formal = RegionsMustUnion(l_formal, 
					    CONS(EFFECT, formal_reg, NIL), 
					    regions_same_action_p);
		regions_free(l_real_exp);
	    }
	    
	} /* else */	
	
    } /* for */
    
    /* il faut calculer l'intersection avec les summary regions de la 
     * fonction pour e'viter certains proble`mes comme avec:
     *
     *      <A(PHI1)-OUT-MUST-{PHI1==I}
     *      CALL TOTO(A(I), A(I))
     *     
     *      <I-R-MUST-{}>, <J-W-MUST-{}
     *      SUBROUTINE TOTO(I,J)
     *
     * si on ne fait pas attention, on obtient <I-OUT-MUST-{}>, <J-OUT-MUST>
     * ve'rifier que c'est compatible avec la norme. Inutile de faire des 
     * choses inutiles. 
     */
    l_formal = RegionsIntersection(l_formal, effects_dup(l_sum_rw_reg),
				   regions_same_action_p);

    return(l_formal);
}


/* static list common_regions_forward_translation
 *                              (entity func, list real_regions)
 * input    : the called function, the list of real arguments at call site, and
 *            the list of regions to translate.
 * output   : the translated list of regions.
 * modifies : nothing.
 * comment  : 
 */
static list common_regions_forward_translation(entity func, list real_regions)
{
    list func_regions = NIL;

    MAP(EFFECT, real_reg,
    {
	storage real_s = entity_storage(region_entity(real_reg));
	/* we are only interested in regions concerning common variables. 
	 * They are  the entities with a ram storagethat are not dynamic
         * variables*/
	if (storage_ram_p(real_s) && 
	    !dynamic_area_p(ram_section(storage_ram(real_s)))
	    && !heap_area_p(ram_section(storage_ram(real_s)))
	    && !stack_area_p(ram_section(storage_ram(real_s))))
	{
	    list regs = common_region_translation(func, real_reg, FORWARD);
	    func_regions = RegionsMustUnion(func_regions, regs, 
					    regions_same_action_p);	
	}
    },
	real_regions);

    return(func_regions);
}

list c_convex_effects_on_actual_parameter_forward_translation
(entity callee, expression real_exp, entity formal_ent, list l_reg, transformer context)
{
  syntax real_s = expression_syntax(real_exp);
  list l_formal = NIL;

  pips_debug_effects(6,"initial regions :\n", l_reg);


  switch (syntax_tag(real_s))
    {
    case is_syntax_call:
      {
	call real_call = syntax_call(real_s);
	entity real_op = call_function(real_call);
	list args = call_arguments(real_call);	
	type uet = ultimate_type(entity_type(real_op));
	value real_op_v = entity_initial(real_op);

	pips_debug(5, "call case, function %s \n", module_local_name(real_op));
	if(type_functional_p(uet)) 
	  {	    
	    if (value_code_p(real_op_v))
	      {
		pips_debug(5, "external function\n");
		pips_user_warning("Nested function calls are ignored. Consider splitting the code before running PIPS\n");
		l_formal = NIL;
		break;
	      }
	    else /* it's an intrinsic */
	      {
		pips_debug(5, "intrinsic function\n");
	     
		if (ENTITY_ASSIGN_P(real_op))
		  {
		    pips_debug(5, "assignment case\n");
		    l_formal = c_convex_effects_on_actual_parameter_forward_translation
		      (callee, EXPRESSION(CAR(CDR(args))), formal_ent, l_reg, context);
		    break;
		  }
		else if(ENTITY_ADDRESS_OF_P(real_op)) 
		  {
		    expression arg1 = EXPRESSION(CAR(args));		    
		    list l_real_arg = NIL;
		    effect eff_real;
		    int nb_phi_real;
		    Psysteme sc_nb_phi_real;
		    expression exp_nb_phi_real = expression_undefined;
		    bool general_case = true;
		    bool in_out = in_out_methods_p();
 
		    pips_debug(5, "address of case\n");
		    
		    /* first we compute a SIMPLE effect on the argument of the address_of operator.
		       This is to distinguish between the general case and the case where
                       the operand of the & operator is an array element.
		       Simple effect indices are easier to retrieve.
		    */
		    set_methods_for_proper_simple_effects();
		    l_real_arg = generic_proper_effects_of_complex_address_expression
		      (arg1, &eff_real, true);
		    nb_phi_real = (int) gen_length(reference_indices(effect_any_reference(eff_real)));
		    gen_full_free_list(l_real_arg);

		    /* there are indices but we don't know if they represent array dimensions, 
		       struct/union/enum fields, or pointer dimensions.
		    */
		    if(nb_phi_real > 0)
		      {
			reference eff_real_ref = effect_any_reference(eff_real);
			list l_inds_real = NIL, l_tmp = NIL; 
			reference ref_tmp;
			type t = type_undefined;
			
			for(l_inds_real = reference_indices(eff_real_ref); !ENDP(CDR(l_inds_real)); POP(l_inds_real))
			  {
			    l_tmp = gen_nconc(l_tmp, CONS(EXPRESSION, copy_expression(EXPRESSION(CAR(l_inds_real))), NIL));
			  }
			
			ref_tmp = make_reference(reference_variable(eff_real_ref), l_tmp);
			t = simple_effect_reference_type(ref_tmp);
			free_reference(ref_tmp);

			if (type_undefined_p(t))
			  pips_internal_error("undefined type not expected \n");

			if(type_variable_p(t) && !ENDP(variable_dimensions(type_variable(t))))
			  {
			    pips_debug(5,"array element or sub-array case\n");
			    general_case = false;
			    /* we build the constraint PHI_nb_phi_real >= last index of eff_real */
			    exp_nb_phi_real = EXPRESSION(CAR(l_inds_real));
			    sc_nb_phi_real = sc_new();
			    (void) sc_add_phi_equation(&sc_nb_phi_real, 
						       copy_expression(exp_nb_phi_real), 
						       nb_phi_real, NOT_EG, NOT_PHI_FIRST);
			  }
			else
			  pips_debug(5, "general case\n");
		      }
		    
		    free_effect(eff_real);
		    eff_real = effect_undefined;
		    /* well, not strictly necessary : forward propagation is only for OUT regions */
		    if (in_out)
		      set_methods_for_convex_in_out_effects();
		    else
		      set_methods_for_convex_rw_effects();
		    init_convex_inout_prettyprint(module_local_name(get_current_module_entity()));
		    
		    /* now we compute a *convex* effect on the argument of the 
		       address_of operator and modify it's last dimension
		       according to the fact that there is an addressing operator
		    */		    

		    l_real_arg = generic_proper_effects_of_complex_address_expression
		      (arg1, &eff_real, true);
		    gen_full_free_list(l_real_arg);
		    
		    if (!general_case)
		      {			
			/* array element operand : we replace the constraint on the last
			   phi variable with */
			entity phi_nb_phi_real = make_phi_entity(nb_phi_real);
			region_exact_projection_along_variable(eff_real, phi_nb_phi_real);
			region_sc_append_and_normalize(eff_real, sc_nb_phi_real, 1); 
			(void) sc_free(sc_nb_phi_real);
		      }
		    
		    FOREACH(EFFECT, eff_orig, l_reg)
		      {
			int nb_phi_orig = (int) gen_length(reference_indices(effect_any_reference(eff_orig)));
			
			/* First we have to test if the eff_real access path leads to the eff_orig access path */
			
			/* to do that, if the entities are the same (well in fact we should also 
			   take care of aliasing), we add the constraints of eff_real to those of eff_orig,
			   and the system must be feasible.
			   We should also take care of linearization here.
			*/ 
			
			if(effect_entity(eff_orig) == effect_entity(eff_real)
			   &&  nb_phi_orig >= nb_phi_real)
			  {
			    effect eff_formal = (*effect_dup_func)(eff_orig);
			    region_sc_append_and_normalize(eff_formal, region_system(eff_real), 1);
			    
			    if (sc_empty_p(region_system(eff_formal)))
			      {
				pips_debug(5, "the original effect does not correspond to the actual argument \n");
				free_effect(eff_formal);
			      }
			    else
			      {
				/* At least part of the original effect corresponds to the actual argument : 
				   we need to translate it 
				*/ 
				Psysteme sc_formal;
				reference ref_formal = effect_any_reference(eff_formal);
				reference new_ref;
				list new_inds = NIL;
				int i, min_phi, min_i;
				
				pips_debug_effect(5, "matching access paths, considered effect is : \n", eff_formal);
				
				/* first we translate the predicate in the callee's name space */
				convex_region_descriptor_translation(eff_formal);		    
				pips_debug_effect(5, "eff_formal after context translation: \n", eff_formal);
				
				/* Then we remove the phi variables common to the two regions
				   except the last one if we are not in the general case */
				/* This is only valid when there is no linearization ; in the general case
				   a translation system should be built
				*/
				sc_formal = region_system(eff_formal);
				for(i = 1; i <= nb_phi_real; i++)
				  {
				    entity phi_i = make_phi_entity(i);
				    entity psi_i = make_psi_entity(i);
				    
				    sc_formal = sc_variable_rename(sc_formal, (Variable) phi_i, (Variable) psi_i);			
				  }
				/* if not in the general case, we add the constraint 
				   phi_nb_phi_real == psi_nb_phi_real - exp_nb_phi_real 
				*/
				if (!general_case)
				  {
				    entity phi = make_phi_entity(nb_phi_real);
				    Pvecteur v_phi = vect_new((Variable) phi, VALUE_ONE);
				    entity psi = make_psi_entity(nb_phi_real);
				    Pvecteur v_psi = vect_new((Variable) psi, VALUE_ONE);
				    Pvecteur v = vect_substract(v_phi, v_psi);
				    normalized nexp = NORMALIZE_EXPRESSION(exp_nb_phi_real);
				    if (normalized_linear_p(nexp)) 
				      {
					pips_debug(6, "normalized last index : "
						   "adding phi_nb_phi_real == psi_nb_phi_real - exp_nb_phi_real \n");
					Pvecteur v1 = vect_copy(normalized_linear(nexp));
					Pvecteur v2;
					v2 = vect_add(v, v1);
					sc_formal = sc_constraint_add(sc_formal, contrainte_make(v2), true);
					vect_rm(v1);
				      }
				    vect_rm(v_psi);
				    vect_rm(v);
				  }
				region_system(eff_formal) = sc_formal;
				pips_debug_effect(5, "eff_formal before removing psi variables: \n", eff_formal);
				region_remove_psi_variables(eff_formal);
				pips_debug_effect(5, "eff_formal after renaming common dimensions: \n", eff_formal);
				
				/* Finally, we must rename remaining phi variables from 2 
				   add a PHI1==0 constraint in the general case, 
				   or, in the contrary, rename remaining phi variables from 1.
				   We must also change the resulting region 
				   entity for the formal entity in all cases.
				*/
				min_phi = general_case? 2:1; 
				min_i = general_case ? nb_phi_real+1 : nb_phi_real;
				sc_formal = region_system(eff_formal);
				
				
				for(i = min_i; i <= nb_phi_orig; i++)
				  {
				    entity phi_i = make_phi_entity(i);
				    entity psi_formal = make_psi_entity(i-min_i+min_phi);
				    
				    sc_formal = sc_variable_rename(sc_formal, (Variable) phi_i, (Variable) psi_formal);
				    
				    new_inds = gen_nconc(new_inds,
							 CONS(EXPRESSION,
							      make_phi_expression(i-nb_phi_real+1),
							      NIL));
				    
				  }
				for(i=min_phi; i<= nb_phi_orig-min_i+min_phi; i++)
				  {
				    entity phi_i = make_phi_entity(i);
				    entity psi_i = make_psi_entity(i);
				    sc_formal = sc_variable_rename(sc_formal, (Variable) psi_i, (Variable) phi_i);
				  }
				region_system(eff_formal) = sc_formal;
				pips_debug_effect(5, "eff_formal after shifting dimensions: \n", eff_formal);

				if(general_case)
				  {
				    /* add PHI1 == 0 */
				    sc_formal = region_system(eff_formal);
				    (void) sc_add_phi_equation(&sc_formal, int_to_expression(0), 1, IS_EG, PHI_FIRST);
				    region_system(eff_formal) = sc_formal;
				    new_inds = CONS(EXPRESSION, make_phi_expression(1), new_inds);				
				  }

				free_reference(ref_formal);
				new_ref = make_reference(formal_ent, new_inds);
				cell_reference(effect_cell(eff_formal)) = new_ref;
				pips_debug_effect(5, "final eff_formal : \n", eff_formal);
				l_formal = RegionsMustUnion(l_formal, CONS(EFFECT, eff_formal, NIL),
							    regions_same_action_p);
				pips_debug_effects(6,"l_formal after adding new effect : \n", l_formal);
				
			      } /* else of the if (sc_empty_p) */
			    			    
			  } /* if(effect_entity(eff_orig) == effect_entity(eff_real) ...)*/
			
		      } /* FOREACH */

		    break;
		  }
		else
		  {
		    pips_debug(5, "Other intrinsic case : entering general case \n");
		  }
	      }	    
	  }
	else if(type_variable_p(uet)) 
	  {
	    pips_user_warning("Effects of call thru functional pointers are ignored\n");
	    l_formal = NIL;
	    break;
	  }
	/* entering general case which includes general calls*/
      }
    case is_syntax_reference:
    case is_syntax_subscript:
      {
	list l_real_arg = NIL;
	effect eff_real;

	pips_debug(5, "general case\n");
	
	/* first we compute an effect on the real_arg */	
	l_real_arg = generic_proper_effects_of_complex_address_expression
	  (real_exp, &eff_real, true);
	gen_full_free_list(l_real_arg);

	FOREACH(EFFECT, eff_orig, l_reg)
	  {
	    int nb_phi_orig = (int) gen_length(reference_indices(effect_any_reference(eff_orig)));
	    int nb_phi_real = (int) gen_length(reference_indices(effect_any_reference(eff_real)));
	    /* First we have to test if the eff_real access path leads to the eff_orig access path */

	    /* to do that, if the entities are the same (well in fact we should also 
	       take care of aliasing), we add the constraints of eff_real to those of eff_orig,
	       and the system must be feasible.
	    */ 

	    if(effect_entity(eff_orig) == effect_entity(eff_real)
	       &&  nb_phi_orig >= nb_phi_real)
	      {
		effect eff_formal = (*effect_dup_func)(eff_orig);
		region_sc_append_and_normalize(eff_formal, region_system(eff_real), 1);
		
		if (sc_empty_p(region_system(eff_formal)))
		  {
		    pips_debug(5, "the original effect does not correspond to the actual argument \n");
		    free_effect(eff_formal);
		  }
		else
		  {
		    /* At least part of the original effect corresponds to the actual argument : 
		       we need to translate it 
		    */ 
		    Psysteme sc_formal;
		    reference ref_formal = effect_any_reference(eff_formal);
		    reference new_ref;
		    list new_inds = NIL;
		    int i;

		    pips_debug_effect(5, "matching access paths, considered effect is : \n", eff_formal);

		    /* first we translate the predicate in the callee's name space */
		    convex_region_descriptor_translation(eff_formal);		    
		    pips_debug_effect(5, "eff_formal after context translation: \n", eff_formal);

		    /* Then we remove the phi variables common to the two regions */
		    /* This is only valid when there is no linearization ; in the general case
		       a translation system should be built
		    */
		    if(nb_phi_real >= 1)
		      {
			sc_formal = region_system(eff_formal);
			for(i = 1; i <= nb_phi_real; i++)
			  {
			    entity phi_i = make_phi_entity(i);
			    entity psi_i = make_psi_entity(i);
			    
			    sc_formal = sc_variable_rename(sc_formal, (Variable) phi_i, (Variable) psi_i);			
			  }
			region_system(eff_formal) = sc_formal;
			region_remove_psi_variables(eff_formal);
			pips_debug_effect(5, "eff_formal after renaming common dimensions: \n", eff_formal);
		      }
		    else
		      pips_debug(8, "nb_phi_real == 0, no need for renaming. \n");

		    /* Finally, we must rename remaining phi variables from 1 
		       and change the resulting region entity for the formal entity
		    */
		    if(nb_phi_real >= 1)
		      {
			sc_formal = region_system(eff_formal);
			for(i = nb_phi_real+1; i <= nb_phi_orig; i++)
			  {
			    entity phi_i = make_phi_entity(i);
			    entity phi_formal = make_phi_entity(i-nb_phi_real);
			    
			    sc_formal = sc_variable_rename(sc_formal, (Variable) phi_i, (Variable) phi_formal);
			    
			    new_inds = gen_nconc(new_inds,
						 CONS(EXPRESSION,
						      make_phi_expression(i-nb_phi_real),
						      NIL));
			    
			  }
			region_system(eff_formal) = sc_formal;
		      }
		    else
		      {
			for (i = 1; i<= nb_phi_orig - nb_phi_real; i++)
			  new_inds = gen_nconc(new_inds,
					       CONS(EXPRESSION,
						    make_phi_expression(i),
						    NIL));  
		      }
		    pips_debug_effect(5, "eff_formal after renaming remaining phi variables: \n", eff_formal);
		    free_reference(ref_formal);
		    new_ref = make_reference(formal_ent, new_inds);
		    cell_reference(effect_cell(eff_formal)) = new_ref;
		    pips_debug_effect(5, "final eff_formal : \n", eff_formal);
		    l_formal = RegionsMustUnion(l_formal, CONS(EFFECT, eff_formal, NIL),regions_same_action_p);
		    pips_debug_effects(6, "l_formal after adding new effect : \n", l_formal);
		  } /* else of the if (sc_empty_p) */


	      } /* if(effect_entity(eff_orig) == effect_entity(eff_real) ...)*/
	      


	    /* */

	  } /* FOREACH */

	break;
      }
    case is_syntax_application:
      {
	pips_internal_error("Application not supported yet\n");
	break;
      }

    case is_syntax_cast:
      {
	pips_debug(6, "cast expression\n");
	l_formal = 
	  c_convex_effects_on_actual_parameter_forward_translation
	  (callee, cast_expression(syntax_cast(real_s)), 
	   formal_ent, l_reg, context);
	break;
      }
    case is_syntax_range:
      {
	pips_user_error("Illegal effective parameter: range\n");
	break;
      }
      
    case is_syntax_sizeofexpression:
      {
	pips_debug(6, "sizeofexpression : -> NIL");
	l_formal = NIL;
	break;
      }
    case is_syntax_va_arg:
      {
	pips_internal_error("va_arg not supported yet\n");
	break;
      }
    default:
      pips_internal_error("Illegal kind of syntax\n");
		
    } /* switch */
  
  
  pips_debug_effects(6,"resulting regions :\n", l_formal);
  return(l_formal);
  
}



/********************************************************* COMMON FUNCTIONS */


/* static list common_region_translation(entity func, region reg, 
 *                                       boolean backward)
 * input    : func is the called function, real_args are the real arguments,
 *            reg is the region to translate (it concerns an array in a common), 
 *            and backward indicates the direction of the translation.
 * output   : a list of regions, that are the translation of the initial region.
 * modifies : nothing: duplicates the original region.
 * comment  : the algorithm is the following
 * 
 * Scan the variables of the common that belong to the target function
 * For each variable do
 *     if it has elements in common with the variable of the initial region
 *        if both variables have the same layout in the common
 *           perform the translation using array_region-translation
 *        else 
 *           use the subscript values, and take into account the relative
 *           offset of the variables in the common
 *           add to the translated region the declaration system of the 
 *           target variable to have a smaller region.
 * until all the elements of the initial variable have been translated.
 */
static list common_region_translation(entity callee, region reg,
				      boolean backward)
{
    list new_regions = NIL;
    entity reg_ent = region_entity(reg);
    entity caller = get_current_module_entity();
    entity source_func = backward ? callee : caller;
    entity target_func = backward ? caller : callee;
    entity entity_target_func = target_func;
    entity ccommon;
    list l_tmp, l_com_ent;
    int reg_ent_size, total_size, reg_ent_begin_offset, reg_ent_end_offset;
    region new_reg;
    boolean found = FALSE;
    

    ifdebug(5)
    {
	pips_debug(5,"input region: \n%s\n", region_to_string(reg));
    }

    /* If the entity is a top-level entity, no translation;
     * It is the case for variables dexcribing I/O effects (LUNS).
     */

    if (top_level_entity_p(reg_ent) || io_entity_p(reg_ent)
	|| rand_effects_entity_p(reg_ent))
    {
	pips_debug(5,"top-level entity.\n");	
	new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     reg_ent, target_func, reference_undefined,
		     0, backward);
	new_regions = CONS(EFFECT, new_reg, NIL);
	return(new_regions);
    }

    

    ifdebug(6)
    {
	pips_debug(5, "target function: %s (local name: %s)\n",
		   entity_name(target_func), module_local_name(target_func));
    }

    /* First, we search if the common is declared in the target function;
     * if not, we have to deterministically choose an arbitrary function
     * in which the common is declared. It will be our reference.
     * By deterministically, I mean that this function shall be chosen whenever
     * we try to translate from this common to a routine where it is not 
     * declared.
     */
    ccommon = ram_section(storage_ram(entity_storage(reg_ent)));
    l_com_ent = area_layout(type_area(entity_type(ccommon)));

    pips_debug(6, "common name: %s\n", entity_name(ccommon));

    for( l_tmp = l_com_ent; !ENDP(l_tmp) && !found; l_tmp = CDR(l_tmp) )
    {
	entity com_ent = ENTITY(CAR(l_tmp));
	if (strcmp(entity_module_name(com_ent),
		   module_local_name(target_func)) == 0)
	{
	    found = TRUE;
	}
    }
    
    /* If common not declared in caller, use the subroutine of the first entity
     * that appears in the common layout. (not really deterministic: I should
     * take the first name in lexical order. BC.
     */
    if(!found)
    {
	entity ent = ENTITY(CAR(l_com_ent));	
	entity_target_func =
	    module_name_to_entity(module_name(entity_name(ent)));
	ifdebug(6)
	{
	    pips_debug(6, "common not declared in caller,\n"
		       "\t using %s declarations instead\n", 
		       entity_name(entity_target_func));
	}
    }

    /* first, we calculate the offset and size of the region entity */
    reg_ent_size = array_size(reg_ent);
    reg_ent_begin_offset = ram_offset(storage_ram(entity_storage(reg_ent)));
    reg_ent_end_offset = reg_ent_begin_offset + reg_ent_size - 1;

    pips_debug(6,
	       "\n\treg_ent: size = %d, offset_begin = %d, offset_end = %d\n",
	       reg_ent_size, reg_ent_begin_offset, reg_ent_end_offset); 

    /* then, we perform the translation */
    ccommon = ram_section(storage_ram(entity_storage(reg_ent)));
    l_com_ent = area_layout(type_area(entity_type(ccommon)));
    total_size = 0;

    for(; !ENDP(l_com_ent) && (total_size < reg_ent_size); 
	l_com_ent = CDR(l_com_ent)) 
    {
	entity new_ent = ENTITY(CAR(l_com_ent));
	
	pips_debug(6, "current entity: %s\n", entity_name(new_ent)); 

	if (strcmp(module_name(entity_name(new_ent)),
		   module_local_name(entity_target_func)) == 0)
	{
	    int new_ent_size = array_size(new_ent);
	    int new_ent_begin_offset = 
		ram_offset(storage_ram(entity_storage(new_ent)));
	    int new_ent_end_offset = new_ent_begin_offset + new_ent_size - 1;

	    pips_debug(6, "\n\t new_ent: size = %d, "
		       "offset_begin = %d, offset_end = %d \n",
		     new_ent_size, new_ent_begin_offset, new_ent_end_offset); 
	    
	    if ((new_ent_begin_offset <= reg_ent_end_offset) && 
		(reg_ent_begin_offset <= new_ent_end_offset ))
		/* these entities have elements in common */
	    {
		int offset = reg_ent_begin_offset - new_ent_begin_offset;
		
		new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     new_ent, target_func, reference_undefined,
		     (Value) offset, backward);
		new_regions = RegionsMustUnion(new_regions,
					       CONS(EFFECT, new_reg, NIL), 
					       regions_same_action_p);
		total_size += min (reg_ent_begin_offset,new_ent_end_offset) 
		    - max(reg_ent_begin_offset, new_ent_begin_offset) + 1;
	    }
	}
    }
	
    ifdebug(5)
    {
	pips_debug(5, "output regions: \n");
	print_regions(new_regions);
    }
    return(new_regions);
}






