/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: utils.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains various useful functions, some of which should be moved
 * elsewhere.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "text-util.h"

#include "effects-generic.h"


/******************************************************** GENERIC FUNCTIONS */

/* GENERIC FUNCTIONS on lists of effects to be instanciated for specific 
   types of effects */

/* initialisation and finalization */
void (*effects_computation_init_func)(string /* module_name */);
void (*effects_computation_reset_func)(string /* module_name */);

/* dup and free - This should be handled by newgen, but there is a problem
 * with the persistency of references - I do not understand what happens. */
effect (*effect_dup_func)(effect eff);
void (*effect_free_func)(effect eff);

/* make functions for effects */
effect (*reference_to_effect_func)(reference, action);
list (*effect_to_store_independent_effect_list_func)(effect, bool);
void (*effect_add_expression_dimension_func)(effect eff, expression exp);
void (*effect_change_ith_dimension_expression_func)(effect eff, expression exp, 
					       int i);

/* union */
effect (*effect_union_op)(effect, effect);
list (*effects_union_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
list (*effects_test_union_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* intersection */
list (*effects_intersection_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* difference */
list (*effects_sup_difference_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
list (*effects_inf_difference_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* composition with transformers */
list (*effects_transformer_composition_op)(list, transformer);
list (*effects_transformer_inverse_composition_op)(list, transformer);

/* composition with preconditions */
list (*effects_precondition_composition_op)(list,transformer);

/* union over a range */
list (*effects_descriptors_variable_change_func)(list, entity, entity);
descriptor (*loop_descriptor_make_func)(loop);
list (*effects_loop_normalize_func)(
    list /* of effects */, entity /* index */, range,
    entity* /* new loop index */, descriptor /* range descriptor */,
    bool /* normalize descriptor ? */);
list (*effects_union_over_range_op)(list, entity, range, descriptor);
descriptor (*vector_to_descriptor_func)(Pvecteur);

/* interprocedural translation */
void (*effects_translation_init_func)(entity /* callee */, 
				      list /* real_args */);
void (*effects_translation_end_func)();
void (*effect_descriptor_interprocedural_translation_op)(effect); 

list (*effects_backward_translation_op)(entity, list, list, transformer);
list (*fortran_effects_backward_translation_op)(entity, list, list, transformer);
list (*effects_forward_translation_op)(entity /* callee */, list /* args */,
				       list /* effects */,
				       transformer /* context */);

list (*c_effects_on_formal_parameter_backward_translation_func)
(list /* of effects */, 
 expression /* args */, 
 transformer /* context */);


/* local to global name space translation */
list (*effects_local_to_global_translation_op)(list);



/* functions to provide context and transformer information */
transformer (*load_context_func)(statement);
transformer (*load_transformer_func)(statement);

bool (*empty_context_test)(transformer);

/* proper to contracted proper effects or to summary effects functions */
effect (*proper_to_summary_effect_func)(effect);

/* normalization of descriptors */
void (*effects_descriptor_normalize_func)(list /* of effects */);

/* getting/putting resources from/to pipsdbm */
statement_effects (*db_get_proper_rw_effects_func)(char *);
void (*db_put_proper_rw_effects_func)(char *, statement_effects);

statement_effects (*db_get_invariant_rw_effects_func)(char *);
void (*db_put_invariant_rw_effects_func)(char *, statement_effects);

statement_effects (*db_get_rw_effects_func)(char *);
void (*db_put_rw_effects_func)(char *, statement_effects);

list (*db_get_summary_rw_effects_func)(char *);
void (*db_put_summary_rw_effects_func)(char *, list);

statement_effects (*db_get_in_effects_func)(char *);
void (*db_put_in_effects_func)(char *, statement_effects);

statement_effects (*db_get_cumulated_in_effects_func)(char *);
void (*db_put_cumulated_in_effects_func)(char *, statement_effects);

statement_effects (*db_get_invariant_in_effects_func)(char *);
void (*db_put_invariant_in_effects_func)(char *, statement_effects);

list (*db_get_summary_in_effects_func)(char *);
void (*db_put_summary_in_effects_func)(char *, list);

list (*db_get_summary_out_effects_func)(char *);
void (*db_put_summary_out_effects_func)(char *, list);

statement_effects  (*db_get_out_effects_func)(char *);
void (*db_put_out_effects_func)(char *, statement_effects);


/*
 * CAUTION! 3 NEXTS ARE OBSOLETE! just kept for the old engine!
 */
/* prettyprint function for debug */
void (*effects_prettyprint_func)(list);

/* prettyprint function for sequential and user views */
text (*effects_to_text_func)(list);
void (*attach_effects_decoration_to_text_func)(text);


/* RESET all generic methods... should be called when pips is started...
 */

#define UNDEF abort

typedef void (*void_function)();
typedef gen_chunk* (*chunks_function)();
typedef list (*list_function)();
typedef bool (*bool_function)();
typedef descriptor (*descriptor_function)();
typedef effect (*effect_function)();
typedef transformer (*transformer_function)();
typedef statement_effects (*statement_effects_function)();
typedef text (*text_function)();

void 
generic_effects_reset_all_methods()
{
    effects_computation_init_func = (void_function) UNDEF;
    effects_computation_reset_func = (void_function) UNDEF;

    effect_dup_func = (effect_function) UNDEF;
    effect_free_func = (void_function) UNDEF;

    effect_union_op = (effect_function) UNDEF;
    effects_union_op = (list_function) UNDEF;
    effects_test_union_op = (list_function) UNDEF;
    effects_intersection_op = (list_function) UNDEF;
    effects_sup_difference_op = (list_function) UNDEF;
    effects_inf_difference_op = (list_function) UNDEF;
    effects_transformer_composition_op = (list_function) UNDEF;
    effects_transformer_inverse_composition_op = (list_function) UNDEF;
    effects_precondition_composition_op = (list_function) UNDEF;
    effects_descriptors_variable_change_func = (list_function) UNDEF;

    effects_loop_normalize_func = (list_function) UNDEF;
    effects_union_over_range_op = (list_function) UNDEF;

    reference_to_effect_func = (effect_function) UNDEF;
    loop_descriptor_make_func = (descriptor_function) UNDEF;
    vector_to_descriptor_func = (descriptor_function) UNDEF;

    effects_backward_translation_op = (list_function) UNDEF;
    effects_forward_translation_op = (list_function) UNDEF;
    effects_local_to_global_translation_op = (list_function) UNDEF;

    load_context_func = (transformer_function) UNDEF;
    load_transformer_func = (transformer_function) UNDEF;
    empty_context_test = (bool_function) UNDEF;
    proper_to_summary_effect_func = (effect_function) UNDEF;
    effects_descriptor_normalize_func = (void_function) UNDEF;

    db_get_proper_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_proper_rw_effects_func = (void_function) UNDEF;
    db_get_invariant_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_invariant_rw_effects_func = (void_function) UNDEF;
    db_get_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_rw_effects_func = (void_function) UNDEF;
    db_get_summary_rw_effects_func = (list_function) UNDEF;
    db_put_summary_rw_effects_func = (void_function) UNDEF;
    db_get_in_effects_func = (statement_effects_function) UNDEF;
    db_put_in_effects_func = (void_function) UNDEF;
    db_get_cumulated_in_effects_func = (statement_effects_function) UNDEF;
    db_put_cumulated_in_effects_func = (void_function) UNDEF;
    db_get_invariant_in_effects_func = (statement_effects_function) UNDEF;
    db_put_invariant_in_effects_func = (void_function) UNDEF;
    db_get_summary_in_effects_func = (list_function) UNDEF;
    db_put_summary_in_effects_func = (void_function) UNDEF;
    db_get_summary_out_effects_func = (list_function) UNDEF;
    db_put_summary_out_effects_func = (void_function) UNDEF;
    db_get_out_effects_func = (statement_effects_function) UNDEF;
    db_put_out_effects_func = (void_function) UNDEF;

    set_contracted_proper_effects(TRUE);
    set_contracted_rw_effects(TRUE);

    set_descriptor_range_p(FALSE);

    /* PRETTYPRINT related functions and settings
     */
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    effects_prettyprint_func = (void_function) UNDEF;
    effects_to_text_func = (text_function) UNDEF;
    attach_effects_decoration_to_text_func = (void_function) UNDEF;

    reset_generic_prettyprints();
}


/********************************************************************* MISC */

/* Statement stack to walk on control flow representation */
DEFINE_GLOBAL_STACK(effects_private_current_stmt, statement)

/* Context stack to keep current context when walking on expressions */
DEFINE_GLOBAL_STACK(effects_private_current_context, transformer)

bool 
effects_private_current_context_stack_initialized_p()
{
    return (effects_private_current_context_stack != stack_undefined);
}

bool 
normalizable_and_linear_loop_p(entity index, range l_range)
{
    Value incr = VALUE_ZERO;
    normalized nub, nlb;
    expression e_incr = range_increment(l_range);
    normalized n;
    bool result = TRUE;
    
    /* Is the loop index an integer variable */
    if (! entity_integer_scalar_p(index))
    {
	pips_user_warning("non integer scalar loop index %s.\n", 
			  entity_local_name(index));
	result = FALSE;	
    }
    else
    {
	/* Is the loop increment numerically known ? */
	n = NORMALIZE_EXPRESSION(e_incr);
	if(normalized_linear_p(n))
	{
	    Pvecteur v_incr = normalized_linear(n);
	    if(vect_constant_p(v_incr)) 
		incr = vect_coeff(TCST, v_incr);	
	}
	
	nub = NORMALIZE_EXPRESSION(range_upper(l_range));
	nlb = NORMALIZE_EXPRESSION(range_lower(l_range));    
	
	result = value_notzero_p(incr) && normalized_linear_p(nub) 
	    && normalized_linear_p(nlb);
    }
    
    return(result);
}

transformer
transformer_remove_variable_and_dup(transformer orig_trans, entity ent)
{
    transformer res_trans = transformer_undefined;

    if (orig_trans != transformer_undefined)
    {
	res_trans = copy_transformer(orig_trans);	
	gen_remove(&transformer_arguments(res_trans), ent);
    }
    return(res_trans);
}


/******************************************* COMBINATION OF APPROXIMATIONS */



/* tag approximation_and(tag t1, tag t2)
 * input    : two approximation tags.
 * output   : the tag representing their "logical and", assuming that 
 *            must = true and may = false.
 * modifies :  nothing 
 */
tag approximation_and(tag t1, tag t2)
{
    if ((t1 == is_approximation_must) && (t2 == is_approximation_must)) 
	return(is_approximation_must);
    else
	return(is_approximation_may);
}


/* tag approximation_or(tag t1, tag t2) 
 * input    : two approximation tags.
 * output   : the tag representing their "logical or", assuming that 
 *            must = true and may = false.
 * modifies : nothing
 */
tag approximation_or(tag t1, tag t2)
{
    if ((t1 == is_approximation_must) || (t2 == is_approximation_must)) 
	return(is_approximation_must);
    else
	return(is_approximation_may);
}

/**************************************** DESCRIPTORS (should not be there) */

static bool descriptor_range_p = FALSE;

void
set_descriptor_range_p(bool b)
{
    descriptor_range_p = b;
}

bool
get_descriptor_range_p(void)
{
    return(descriptor_range_p);
}

descriptor
descriptor_inequality_add(descriptor d, Pvecteur v)
{
    if (!VECTEUR_UNDEFINED_P(v))
    {
	Psysteme sc = descriptor_convex(d);
	Pcontrainte contrainte = contrainte_make(v);
	sc_add_inegalite(sc, contrainte);   
	sc->base = BASE_NULLE;
	sc_creer_base(sc);
	descriptor_convex_(d) = newgen_Psysteme(sc);
    }
    return d;
}

transformer
descriptor_to_context(descriptor d)
{
    transformer context;
    if (descriptor_none_p(d))
	context = transformer_undefined;
    else
    {
	Psysteme sc = sc_dup(descriptor_convex(d));
	context = make_transformer(NIL, make_predicate(sc));
    }
    return(context);
}

void
descriptor_variable_rename(descriptor d, entity old_ent, entity new_ent)
{
    if (descriptor_convex_p(d))
    {
	sc_variable_rename(descriptor_convex(d), 
			   (Variable) old_ent, 
			   (Variable) new_ent);
    }
}

descriptor
descriptor_append(descriptor d1, descriptor d2)
{
    if (descriptor_convex_p(d1) && descriptor_convex_p(d2))
    {
	Psysteme
	    sc1 = descriptor_convex(d1),
	    sc2 = descriptor_convex(d2);
	
	sc1 = sc_safe_append(sc1, sc2);
	descriptor_convex_(d1) = newgen_Psysteme(sc1);
    }	
    else
	d1 = descriptor_undefined;
    return d1;
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/

transformer
load_undefined_context(statement s __attribute__ ((__unused__)) )
{
    return transformer_undefined;
}

transformer
load_undefined_transformer(statement s __attribute__ ((__unused__)) )
{
    return transformer_undefined;
}

bool
empty_context_test_false(transformer context __attribute__ ((__unused__)) )
{
    return FALSE;
}

void 
effects_computation_no_init(string module_name __attribute__ ((__unused__)) )
{
    return;
}

void 
effects_computation_no_reset(string module_name __attribute__ ((__unused__)) )
{
    return;
}


/***********************************************************************/
/* FORMER effects/utils.c                                              */
/***********************************************************************/


string vect_debug_entity_name(e)
entity e;
{
    return((e == (entity) TCST) ? "TCST" : entity_name(e));
}

/* bool integer_scalar_read_effects_p(cons * effects): checks that *all* read
 * effects in effects are on integer scalar variable
 *
 * Francois Irigoin: I do not see where the "read" type is checked FI
 */
bool integer_scalar_read_effects_p(fx)
cons * fx;
{
    MAPL(ceffect,
     {entity e = 
	  reference_variable(effect_any_reference(EFFECT(CAR(ceffect))));
     if(!integer_scalar_entity_p(e)) return FALSE;},
	 fx);
    return TRUE;
}

/* check that *some* read or write effects are on integer variables
 *
 * FI: this is almost always true because of array subscript expressions
 */
bool some_integer_scalar_read_or_write_effects_p(fx)
cons * fx;
{
    MAPL(ceffect,
     {entity e = 
	  reference_variable(effect_any_reference(EFFECT(CAR(ceffect))));
	  if(integer_scalar_entity_p(e)) return TRUE;},
	 fx);
    return FALSE;
}

/* bool effects_write_entity_p(cons * effects, entity e): check whether e
 * is written by effects "effects" or not
 */
bool effects_write_entity_p(fx, e)
cons * fx;
entity e;
{
    bool write = FALSE;
    MAP(EFFECT, ef, 
    {
	action a = effect_action(ef);
	entity e_used = reference_variable(effect_any_reference(ef));
	
	/* Note: to test aliasing == should be replaced below by
	 * entity_conflict_p()
	 */
	if(e==e_used && action_write_p(a)) {
	    write = TRUE;
	    break;
	}
    },
	fx);
    return write;
}

/* bool effects_read_or_write_entity_p(cons * effects, entity e): check whether e
 * is read or written by effects "effects" or not accessed at all
 */
bool effects_read_or_write_entity_p(fx, e)
cons * fx;
entity e;
{
    bool read_or_write = FALSE;
    MAPL(cef, 
     {
	 effect ef = EFFECT(CAR(cef));
	 entity e_used = reference_variable(effect_any_reference(ef));
	 /* Used to be a simple pointer equality test */
	 if(entity_conflict_p(e, e_used)) {
	     read_or_write = TRUE;
	     break;
	 }
     },
	 fx);
    return read_or_write;
}

entity effects_conflict_with_entity(fx, e)
cons * fx;
entity e;
{
    entity conflict_e = entity_undefined;
    MAPL(cef, 
     {
	 effect ef = EFFECT(CAR(cef));
	 entity e_used = reference_variable(effect_any_reference(ef));
	 if(entity_conflict_p(e, e_used)) {
	     conflict_e = e_used;
	     break;
	 }
     },
	 fx);
    return conflict_e;
}

list effects_conflict_with_entities(fx, e)
cons * fx;
entity e;
{
    list lconflict_e = NIL;
    MAPL(cef, 
     {
	 effect ef = EFFECT(CAR(cef));
	 entity e_used = reference_variable(effect_any_reference(ef));
	 if(entity_conflict_p(e, e_used)) {
	     lconflict_e = gen_nconc(lconflict_e, 
				     CONS(ENTITY, e_used, NIL));
	    
	 }
     },
	 fx);
    return lconflict_e;
}


/*************** I/O EFFECTS *****************/
bool io_effect_entity_p(entity e)
{
    return io_entity_p(e) && 
	same_string_p(entity_local_name(e), IO_EFFECTS_ARRAY_NAME);
}

/* Return true if a statement has an I/O effect in the effects
   list. */
bool
statement_io_effect_p(statement s)
{
   bool io_effect_found = FALSE;
   list effects_list = load_proper_rw_effects_list(s);

   /* If there is an I/O effects, the following entity should
      exist. If it does not exist, statement_io_effect_p() will return
      FALSE anyway. */
   entity private_io_entity =
      global_name_to_entity(IO_EFFECTS_PACKAGE_NAME,
                            IO_EFFECTS_ARRAY_NAME);

   MAP(EFFECT, an_effect,
       {
          reference a_reference = effect_any_reference(an_effect);
          entity a_touched_variable =
             reference_variable(a_reference);

          if (a_touched_variable == private_io_entity) {
             io_effect_found = TRUE;
             break;
          }
       },
       effects_list);

   return io_effect_found;
}

/* Return TRUE if the statement has a write effect on at least one of
   the argument (formal parameter) of the module and if the argument
   passing mode is by reference. Note that the return variable of a
   function is also considered here as a formal parameter. */
bool statement_has_a_formal_argument_write_effect_p(statement s)
{
   bool write_effect_on_a_module_argument_found = FALSE;
   entity module = get_current_module_entity();
   list effects_list = load_proper_rw_effects_list(s);
   /* it might be better to check the parameter passing mode itself,
      via the module type */
   bool fortran_p = fortran_module_p(module);

   FOREACH(EFFECT, an_effect, effects_list) {
     entity a_variable = reference_variable(effect_any_reference(an_effect));
     bool formal_p = variable_is_a_module_formal_parameter_p(a_variable,
							module);
     bool return_variable_p = variable_return_p(a_variable);

     if (action_write_p(effect_action(an_effect))
	 && (return_variable_p
	     || (formal_p && fortran_p)
	     )
	 ) {
       write_effect_on_a_module_argument_found = TRUE;
       break;
     }
   }       ;

   return write_effect_on_a_module_argument_found;

}



list /* of effect */ make_effects_for_array_declarations(list refs)
{     
  list leff = NIL;
  effect eff;
  MAPL(l1,
  {  
    
    reference ref = REFERENCE(CAR(l1));
    /* FI: in this context, I assume that eff is never returned undefined */
     eff = (*reference_to_effect_func)(ref,make_action(is_action_read, UU));
     if(effect_undefined_p(eff)) {
       pips_debug(8, "Reference to \"%s\" ignored\n", entity_name(reference_variable(ref)));
     }
     else
       leff= CONS(EFFECT,eff,leff);
  },refs);
  
  
  gen_free_list(refs);
  return leff;
}

     

typedef struct { list le, lr; } deux_listes;

static void make_uniq_reference_list(reference r, deux_listes * l)
{
  entity e = reference_variable(r);
  if (! (storage_rom_p(entity_storage(e)) && 
	 !(value_undefined_p(entity_initial(e))) &&
	 value_symbolic_p(entity_initial(e)) &&
	 type_functional_p(entity_type(e)))) {
    
    /* Add reference r only once */
    if (l->le ==NIL || !gen_in_list_p(e, l->le)) {
      l->le = CONS(ENTITY,e,  l->le); 
      l->lr = CONS(REFERENCE,r,l->lr);
    }
  }
}

/* FI: this function has not yet been extended for C types!!! */
list extract_references_from_declarations(list decls)
{
  list arrays = NIL; 
  deux_listes lref = { NIL, NIL };
  
  MAPL(le,{ 
    entity e= ENTITY(CAR(le));
    type t = entity_type(e);
    
    if (type_variable_p(t) && !ENDP(variable_dimensions(type_variable(t))))
      arrays = CONS(VARIABLE,type_variable(t), arrays);
  }, decls );
  
  MAPL(array,
  { variable v = VARIABLE(CAR(array));
  list ldim = variable_dimensions(v);  
  while (!ENDP(ldim))
    {
      dimension d = DIMENSION(CAR(ldim)); 
      gen_context_recurse(d, &lref, reference_domain, make_uniq_reference_list, gen_null);
      ldim=CDR(ldim);
      
    }	   
  }, arrays);
  gen_free_list(lref.le);
  
  return(lref.lr);
}   

list summary_effects_from_declaration(string module_name __attribute__ ((unused)))
{
  list sel = NIL;
  //entity mod = module_name_to_entity(module_name);
  //list decls = code_declarations(value_code(entity_initial(mod)));
  extern list current_module_declarations(void);
  list decls = current_module_declarations();
  list refs = list_undefined;

  refs = extract_references_from_declarations(decls);

  ifdebug(8) {
    pips_debug(8, "References from declarations:\n");
    MAP(REFERENCE, r, {
      pips_debug(8, "Reference for variable \"%s\"\n",
		 entity_name(reference_variable(r)));
      print_reference(r);
      fprintf(stderr, "\n");
    }, refs);
  }

  sel = make_effects_for_array_declarations(refs);

  return sel;
}

void dump_cell(cell c)
{
  fprintf(stderr, "Cell %p = (cell_tag=%td, reference=%p)\n", c, cell_tag(c),
	  cell_preference_p(c)? preference_reference(cell_preference(c)):cell_reference(c));
}

void dump_effect(effect e)
{
  cell c = effect_cell(e);
  action ac = effect_action(e);
  approximation ap = effect_approximation(e);
  descriptor d = effect_descriptor(e);

  effect_consistent_p(e);
  fprintf(stderr, "Effect %p = (domain=%td, cell=%p, action=%p,"
	  " approximation=%p, descriptor=%p\n",
	  e, effect_domain_number(e), c, ac, ap, d);
  dump_cell(c);
}

void dump_effects(list le)
{
  int i = 1;
  MAP(EFFECT, e, {
      fprintf(stderr, "%d ", i++);
      dump_effect(e);
    }, le);
}

/* Check if a reference appears more than once in the effect list. If
   persistant_p is true, do not go thru persistant arcs. Else, use all
   references. */
bool effects_reference_sharing_p(list el, bool persistant_p)
{
  bool sharing_p = FALSE;
  list rl = NIL; /* reference list */
  list ce = list_undefined; /* current effect */

  for(ce=el; !ENDP(ce); POP(ce)) {
    effect e = EFFECT(CAR(ce));
    cell c = effect_cell(e);
    reference r = reference_undefined;

    if(persistant_p) {
      if(cell_reference_p(c))
	r = cell_reference(c);
    }
    else
      r = effect_any_reference(e);

    if(!reference_undefined_p(r)) {
      if(gen_in_list_p((void *) r, rl)) {
	extern void print_effect(effect);
	fprintf(stderr, "this effect shares its reference with another effect in the list\n");
	print_effect(e);
	sharing_p = TRUE;
	break;
      }
      else
	rl = CONS(REFERENCE, r, rl);
    }
  }
  return sharing_p;
}

/************************ anywhere effects ********************/

/**
 @return a new anywhere effect.
 @param ac is an action which is directly used in the new effect

 Allocate a new anywhere effect using generic function 
 reference_to_effect_func, and the anywhere entity on demand
 which may not be best if we want to express it's aliasing with all
 module areas. In the later case, the anywhere entity should be
 generated by bootstrap and be updated each time new areas are
 declared by the parsers. I do not use a persistant anywhere
 reference to avoid trouble with convex-effect nypassing of the
 persistant pointer. (re-used from original non-generic function
 anywhere_effect.)
 
   Action a is integrated in the new effect (aliasing).
 */
effect make_anywhere_effect(action ac)
{
 
  entity anywhere_ent = gen_find_tabulated(ALL_MEMORY_ENTITY_NAME, 
					   entity_domain);
  effect anywhere_eff = effect_undefined;

  if(entity_undefined_p(anywhere_ent)) 
    {
      area a = make_area(0,NIL); /* Size and layout are unknown */
      type t = make_type_area(a);
      anywhere_ent = make_entity(strdup(ALL_MEMORY_ENTITY_NAME),
				 t, MakeStorageRom(), MakeValueUnknown());
    }
  
  anywhere_eff = (*reference_to_effect_func)
    (make_reference(anywhere_ent, NIL),
     ac);
  effect_to_may_effect(anywhere_eff);
  return anywhere_eff;
  
}


/********************** Effects on all accessible paths  ***************/

/**
  @param eff_write a write effect
  @param is the action of the generated effects :
         'r' for read, 'w' for write, and 'x' for read and write.
  @return a list of effects. beware : eff_write is included in the list.

 */
list effect_to_effects_with_given_tag(effect eff, tag act)
{
  list l_res = NIL;
  effect eff_read = effect_undefined;
  effect eff_write = effect_undefined;
  extern void print_effect(effect);
	
  pips_assert("effect is defined \n", !effect_undefined_p(eff));

  if (act == 'x')
    {      
      eff_write = eff;
      effect_action_tag(eff_write) = is_action_write;
      eff_read = copy_effect(eff_write);
      effect_action_tag(eff_read) = is_action_read;
    }
  else if (act == 'r')
    {
      
      eff_read = eff;
      effect_action_tag(eff_read) = is_action_read;
      eff_write = effect_undefined;
    }
  else
    {
      eff_read = effect_undefined;
      eff_write = eff;
      effect_action_tag(eff_write) = is_action_write;
    }
  
  ifdebug(8)
    {
      pips_debug(8, "adding effects to l_res : \n");
      if(!effect_undefined_p(eff_write)) 
	print_effect(eff_write);
      if(!effect_undefined_p(eff_read)) 
	print_effect(eff_read);
    }
  
  if(!effect_undefined_p(eff_write)) 
    l_res = gen_nconc(l_res, CONS(EFFECT, eff_write, NIL));
  if(!effect_undefined_p(eff_read)) 
    l_res = gen_nconc(l_res, CONS(EFFECT, eff_read, NIL)); 

  return l_res;
}

/**
   @param eff is an effect whose reference is the beginning access path.
          it is not modified or re-used.
   @param eff_type is the type of the object represented by the effect
          access path. This avoids computing it at each step.
   @param act is the action of the generated effects :
              'r' for read, 'w' for write, and 'x' for read and write.
   @return a list of effects on all the accessible paths from eff reference.
 */
list generic_effect_generate_all_accessible_paths_effects(effect eff, 
							  type eff_type, 
							  tag act)
{
  list l_res = NIL;
  pips_assert("the effect must be defined\n", !effect_undefined_p(eff));
  extern void print_effect(effect);
	
  
  if (anywhere_effect_p(eff))
    {
      /* there is no other accessible path */
      pips_debug(6, "anywhere effect -> returning NIL \n");
      
    }
  else
    {
      reference ref = effect_any_reference(eff);
      entity ent = reference_variable(ref);
      type t = ultimate_type(entity_type(ent));
      int d = effect_type_depth(t);
      effect eff_write = effect_undefined;

      /* this may lead to memory leak if no different access path is 
	 reachable */
      eff_write = copy_effect(eff);
      
      ifdebug(6)
	{
	  pips_debug(6, "considering effect : \n");
	  print_effect(eff);
	  pips_debug(6, " with entity effect type depth %d \n",
		     d);
	}
      
           
      switch (type_tag(eff_type))
	{
	case is_type_variable :
	  {
	    variable v = type_variable(eff_type);
	    basic b = variable_basic(v);
	    bool add_effect = false;
	    
	    pips_debug(8, "variable case, of dimension %d\n", 
		       (int) gen_length(variable_dimensions(v))); 

	    /* we first add the array dimensions if any */
	    FOREACH(DIMENSION, c_t_dim, 
		    variable_dimensions(v))
	      {
		(*effect_add_expression_dimension_func)
		  (eff_write, make_unbounded_expression());
		add_effect = true;
	      }
	    /* And add the generated effect */
	    if (add_effect)
	      {
		l_res = gen_nconc
		  (l_res,
		   effect_to_effects_with_given_tag(eff_write,act));
		add_effect = false;
	      }

	    /* If the basic is a pointer type, we must add an effect
	       with a supplementary dimension, and then recurse
               on the pointed type.
	    */
	    if(basic_pointer_p(b))
	      {
		pips_debug(8, "pointer case, \n");
				
		eff_write = copy_effect(eff_write);
		(*effect_add_expression_dimension_func)
		  (eff_write, make_unbounded_expression());
		
		l_res = gen_nconc
		  (l_res,
		   effect_to_effects_with_given_tag(eff_write,act));
		
		l_res = gen_nconc
		  (l_res,
		   generic_effect_generate_all_accessible_paths_effects
		   (eff_write,  basic_pointer(b), act));
		
	      }	    	    
	    
	    break;
	  }
	default:
	  {
	    pips_internal_error("case not handled yet\n");
	  }
	} /*switch */
      
    } /* else */
  
  
  return(l_res);
}

/******************************************************************/

static list type_fields(type t)
{
  list l_res = NIL;

  switch (type_tag(t))
    {
    case is_type_struct:
      l_res = type_struct(t);
      break;
    case is_type_union:
      l_res = type_union(t);
      break;
    case is_type_enum:
      l_res = type_enum(t);
    default:
      pips_internal_error("type_fields improperly called\n");
    }
  return l_res;
	
}

/** 
 NOT YET IMPLEMENTED FOR VARARGS AND FUNCTIONAL TYPES.

 @param eff is an effect
 @return true if the effect reference maybe an access path to a pointer
*/
static bool r_effect_pointer_type_p(effect eff, list l_ind, type ct)
{
  bool p = false, finished = false;

  pips_debug(7, "begin with type %s\n and number of indices : %d\n", 
	     words_to_string(words_type(ct)),
	     (int) gen_length(l_ind));
  while (!finished)
    {
      switch (type_tag(ct))
	{
	case is_type_variable :
	  {
	    variable v = type_variable(ct);
	    basic b = variable_basic(v);
	    list l_dim = variable_dimensions(v);
	    
	    pips_debug(8, "variable case, of dimension %d\n", 
		       (int) gen_length(variable_dimensions(v))); 

	    while (!ENDP(l_dim) && !ENDP(l_ind))
	      {
		POP(l_dim);
		POP(l_ind);
	      }
	    
	    if(ENDP(l_ind) && ENDP(l_dim))
	      {
	      if(basic_pointer_p(b))
		{
		  p = true;		  
		  finished = true;
		}
	      else
		finished = true;
	      }
	    else if (ENDP(l_dim)) /* && !ENDP(l_ind) by construction */
	      {
		pips_assert("the current basic should be a pointer or a derived\n", 
			    basic_pointer_p(b) || basic_derived_p(b));	
		
		if (basic_pointer_p(b))
		  {
		    ct = basic_pointer(b);
		    POP(l_ind);
		  }
		else /* b is a derived */ 
		  {
		    ct = entity_type(basic_derived(b));	
		    p = r_effect_pointer_type_p(eff, l_ind, ct);
		    finished = true;
		  }
		 
	      }	
	    else /* ENDP(l_ind) but !ENDP(l_dim) */
	      {
		finished = true;
	      }
	    
	    break;
	  }
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  {
	    list l_ent = type_fields(ct);
	    expression rank_exp = EXPRESSION(CAR(l_ind));
	    int rank;
	    
	    pips_debug(7, "field case, with rank expression : %s \n",
		       words_to_string(words_expression(rank_exp)));

	    /* If the field rank is known, we only look at the 
	       corresponding type.
	       If not, we have to recursively look at each field 
	    */ 
	    if (expression_integer_value(rank_exp, &rank))
	      {
		/* the current type becomes the type of the 
		 *p_rank-th field
		 */		
		ct = entity_type(ENTITY(gen_nth(rank - 1, l_ent)));
		p = r_effect_pointer_type_p(eff, CDR(l_ind), ct);
		finished = true;
	      }
	    else
	      /* look at each field until a pointer is found*/
	      {
		while (!ENDP(l_ent) && p)
		  {
		    type new_ct = entity_type(ENTITY(CAR(l_ent)));
		    p = r_effect_pointer_type_p(eff, CDR(l_ind), 
						new_ct);
		    POP(l_ent);
		  }
		finished = true;
	      }	    
	    break;
	  }
	default:
	  {
	    pips_internal_error("case not handled yet\n");
	  }
	} /*switch */

    }/*while */
  pips_debug(8, "end with p = %s\n", p== false ? "false" : "true");
  return p;

}


/** 
 NOT YET IMPLEMENTED FOR VARARGS AND FUNCTIONAL TYPES.

 @param eff is an effect
 @return true if the effect reference maybe an access path to a pointer
*/
bool effect_pointer_type_p(effect eff)
{
  bool p = false;
  reference ref = effect_any_reference(eff);
  list l_ind = reference_indices(ref);
  entity ent = reference_variable(ref);
  type t = basic_concrete_type(entity_type(ent));

  pips_debug(8, "begin with effect reference %s\n",
	     words_to_string(words_reference(ref)));

  p = r_effect_pointer_type_p(eff, l_ind, t);

  pips_debug(8, "end with p = %s\n", p== false ? "false" : "true");
  return p;

}


bool regions_weakly_consistent_p(list rl)
{
  FOREACH(EFFECT, r, rl) {
    descriptor rd = effect_descriptor(r);

    if(descriptor_convex_p(rd)) {
      Psysteme rsc = descriptor_convex(rd);

      pips_assert("rsc is weakly consistent", sc_weak_consistent_p(rsc));
    }
  }
  return TRUE;
}

bool region_weakly_consistent_p(effect r)
{
  descriptor rd = effect_descriptor(r);

  if(descriptor_convex_p(rd)) {
    Psysteme rsc = descriptor_convex(rd);

    pips_assert("rsc is weakly consistent", sc_weak_consistent_p(rsc));
  }

  return TRUE;
}

/**
   Effects are not copied but a new list is built.
 */
list statement_modified_pointers_effects_list(statement s)
{
  list l_cumu_eff = load_rw_effects_list(s);
  list l_res = NIL;
  bool anywhere_p = false;

  ifdebug(6){
	 pips_debug(6, " effects before selection: \n");
	 (*effects_prettyprint_func)(l_cumu_eff);
       }

  FOREACH(EFFECT, eff, l_cumu_eff)
    {
      if (!anywhere_p && effect_write_p(eff))
	{
	  if (anywhere_effect_p(eff))
	    anywhere_p = true;
	  else if (effect_pointer_type_p(eff))
	    l_res = gen_nconc(l_res, CONS(EFFECT, eff, NIL));
	}
    }
  if (anywhere_p)
    {
      gen_free_list(l_res);
      l_res = CONS(EFFECT, make_anywhere_effect(make_action_write()), NIL);
    }

  ifdebug(6){
	 pips_debug(6, " effects after selection: \n");
	 (*effects_prettyprint_func)(l_res);
       }


  return l_res;
}

/******************************************************************/

list generic_effects_store_update(list l_eff, statement s, bool backward_p)
{

   transformer t; /* transformer of statement s */
   list l_eff_pointers, l_eff_tmp;
   list l_res = NIL;
   bool anywhere_w_p = false;
   bool anywhere_r_p = false;

   pips_debug(5, "begin\n");
	
   t = (*load_transformer_func)(s);    

   if (l_eff !=NIL)    
     {
       /* first change the store of the descriptor */
       if (backward_p)
	 l_eff = (*effects_transformer_composition_op)(l_eff, t);
       else
	 l_eff =  (*effects_transformer_inverse_composition_op)(l_eff, t);
   
       ifdebug(5){
	 pips_debug(5, " effects after composition with transformer: \n");
	 (*effects_prettyprint_func)(l_eff);
       }

       if (get_bool_property("EFFECTS_POINTER_MODIFICATION_CHECKING"))
	 {
	   /* then change the effects references if some pointer is modified */
	   /* backward_p is not used here because we lack points_to information
	      and we thus generate anywhere effects 
	   */
	   l_eff_pointers = statement_modified_pointers_effects_list(s);
	   
	   while( !ENDP(l_eff) && 
		  ! (anywhere_w_p && anywhere_r_p))
	     {
	       list l_eff_p_tmp = l_eff_pointers;
	       effect eff = EFFECT(CAR(l_eff));
	       bool eff_w_p = effect_write_p(eff);
	       bool found = false;
	       
	       
	       
	       while( !ENDP(l_eff_p_tmp) &&
		      !((eff_w_p && anywhere_w_p) || (!eff_w_p && anywhere_r_p)))
		 {
		   effect eff_p = EFFECT(CAR(l_eff_p_tmp));
		   reference eff_ref = effect_any_reference(eff);
		   reference eff_ref_p = effect_any_reference(eff_p);
		   effect new_eff = effect_undefined;
		   
		   if (same_entity_p(reference_variable(eff_ref), 
				     reference_variable(eff_ref_p)))
		     {
		       /* this is a very rough approximation ! */
		       if (gen_length(reference_indices(eff_ref_p)) <=
			   gen_length(reference_indices(eff_ref)))
			 {
			   new_eff = make_anywhere_effect
			     (copy_action(effect_action(eff)));
			   l_res = gen_nconc(l_res, CONS(EFFECT, new_eff, NIL));
			   found = true;
			   if (eff_w_p)
			     anywhere_w_p = true;
			   else 
			     anywhere_r_p = true;
			   
			 }
		     } /* if(same_entity_p()) */
		   
		   POP(l_eff_p_tmp);
		 } /* while( !ENDP(l_eff_p_tmp))*/ 
	       
	       /* if we have found no modifiying pointer, we keep the effect */
	       if (!found)
		 {
		   /* is the copy necessary ?*/
		   l_res = gen_nconc(l_res, CONS(EFFECT,copy_effect(eff) , NIL));
		   
		 }
	       
	       POP(l_eff);
	       
	     } /* while( !ENDP(l_eff)) */
	   
	   ifdebug(5){
	     pips_debug(5, " effects after composition with pointer effects: \n");
	     (*effects_prettyprint_func)(l_res);
	   }
            
	 } /* if (get_bool_property("EFFECTS_POINTER_MODIFICATION_CHECKING"))*/
       else
	 l_res = l_eff;
     } /* if (l_eff !=NIL) */

   return l_res;
}
