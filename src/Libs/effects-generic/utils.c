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
list (*effects_backward_translation_op)(entity, list, list, transformer);
list (*effects_forward_translation_op)(entity /* callee */, list /* args */,
				       list /* effects */,
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

void 
generic_effects_reset_all_methods()
{
    effects_computation_init_func = (void_function) UNDEF;
    effects_computation_reset_func = (void_function) UNDEF;

    effect_dup_func = (chunks_function) UNDEF;
    effect_free_func = (void_function) UNDEF;

    effect_union_op = (chunks_function) UNDEF;
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

    reference_to_effect_func = (chunks_function) UNDEF;
    loop_descriptor_make_func = (chunks_function) UNDEF;
    vector_to_descriptor_func = (chunks_function) UNDEF;

    effects_backward_translation_op = (list_function) UNDEF;
    effects_forward_translation_op = (list_function) UNDEF;
    effects_local_to_global_translation_op = (list_function) UNDEF;

    load_context_func = (chunks_function) UNDEF;
    load_transformer_func = (chunks_function) UNDEF;
    empty_context_test = (bool_function) UNDEF;
    proper_to_summary_effect_func = (descriptor_function) UNDEF;
    effects_descriptor_normalize_func = (void_function) UNDEF;

    db_get_proper_rw_effects_func = (chunks_function) UNDEF;
    db_put_proper_rw_effects_func = (void_function) UNDEF;
    db_get_invariant_rw_effects_func = (chunks_function) UNDEF;
    db_put_invariant_rw_effects_func = (void_function) UNDEF;
    db_get_rw_effects_func = (chunks_function) UNDEF;
    db_put_rw_effects_func = (void_function) UNDEF;
    db_get_summary_rw_effects_func = (list_function) UNDEF;
    db_put_summary_rw_effects_func = (void_function) UNDEF;
    db_get_in_effects_func = (chunks_function) UNDEF;
    db_put_in_effects_func = (void_function) UNDEF;
    db_get_cumulated_in_effects_func = (chunks_function) UNDEF;
    db_put_cumulated_in_effects_func = (void_function) UNDEF;
    db_get_invariant_in_effects_func = (chunks_function) UNDEF;
    db_put_invariant_in_effects_func = (void_function) UNDEF;
    db_get_summary_in_effects_func = (list_function) UNDEF;
    db_put_summary_in_effects_func = (void_function) UNDEF;
    db_get_summary_out_effects_func = (list_function) UNDEF;
    db_put_summary_out_effects_func = (void_function) UNDEF;
    db_get_out_effects_func = (chunks_function) UNDEF;
    db_put_out_effects_func = (void_function) UNDEF;

    set_contracted_proper_effects(TRUE);
    set_contracted_rw_effects(TRUE);

    set_descriptor_range_p(FALSE);

    /* PRETTYPRINT related functions and settings
     */
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    effects_prettyprint_func = (void_function) UNDEF;
    effects_to_text_func = (chunks_function) UNDEF;
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
load_undefined_context(statement s)
{
    return transformer_undefined;
}

transformer
load_undefined_transformer(statement s)
{
    return transformer_undefined;
}

bool
empty_context_test_false(transformer context)
{
    return FALSE;
}

void 
effects_computation_no_init(string module_name)
{
    return;
}

void 
effects_computation_no_reset(string module_name)
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
	  reference_variable(effect_reference(EFFECT(CAR(ceffect))));
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
	  reference_variable(effect_reference(EFFECT(CAR(ceffect))));
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
	entity e_used = reference_variable(effect_reference(ef));
	
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
	 entity e_used = reference_variable(effect_reference(ef));
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
	 entity e_used = reference_variable(effect_reference(ef));
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
	 entity e_used = reference_variable(effect_reference(ef));
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
          reference a_reference = effect_reference(an_effect);
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
   the argument (formal parameter) of the module. Note that the return
   variable of a function is also considered here as a formal
   parameter. */
bool
statement_has_a_formal_argument_write_effect_p(statement s)
{
   bool write_effect_on_a_module_argument_found = FALSE;
   entity module = get_current_module_entity();
   list effects_list = load_proper_rw_effects_list(s);

   MAP(EFFECT, an_effect,
       {
          entity a_variable = reference_variable(effect_reference(an_effect));
          
          if (action_write_p(effect_action(an_effect))
              && (variable_return_p(a_variable)
		  || variable_is_a_module_formal_parameter_p(a_variable,
							     module))) {
	      write_effect_on_a_module_argument_found = TRUE;
             break;
          }
       },
       effects_list);

   return write_effect_on_a_module_argument_found;

}

