/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: binary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains generic binary operators for effects and lists of them.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "effects-generic.h"


/************************************************* GENERIC BINARY OPERATORS */

/* list list_of_effects_generic_binary_op(list l1,l2,
 *                         bool (*r1_r2_combinable_p)(),
 *                         list (*r1_r2_binary_op)(),
 *                         list (*r1_unary_op)(),
 *                         list (*r2_unary_op)())
 * input : two lists of effects ; a boolean function that takes two
 *         individual effects as arguments and renders TRUE when they are
 *         considered as combinable ; a binary operator that combines two
 *         individual effects; two unary operators that deal with the
 *         remnants of the two initial lists; these remnants are the effects
 *         that are not combinable with any effect of the other list.
 * output : a list of effects, combination of l1 and l2
 * modifies : l1 and l2, and their effects.
 * comment : ?
*/
list
list_of_effects_generic_binary_op(
    list l1,
    list l2,
    bool (*r1_r2_combinable_p)(effect,effect),
    list (*r1_r2_binary_op)(effect,effect),
    list (*r1_unary_op)(effect),
    list (*r2_unary_op)(effect))
{
    list l_res = NIL;

    debug_on("EFFECTS_OPERATORS_DEBUG_LEVEL");

    ifdebug(1)
    {
      pips_debug(1, "Initial effects : \n");
      fprintf(stderr,"\t l1 :\n");
      (*effects_prettyprint_func)(l1);
      fprintf(stderr,"\t l2 :\n");
      (*effects_prettyprint_func)(l2);
    }
    
    /* we first deal with the effects of l1 : those that are combinable with 
     * the effects of l2, and the others, which we call the remnants of l1 */
    MAP(EFFECT, r1,
    {
      list lr2 = l2;
      list prec_lr2 = NIL;
      bool combinable = FALSE;
      
      pips_debug(8, "r1: %s\n", entity_name(effect_variable(r1)));
      
      while(!combinable && !ENDP(lr2))
      {
	effect r2 = EFFECT(CAR(lr2));
	
	pips_debug(8, "r2: %s\n", entity_name(effect_variable(r2)));
	
	if ( (*r1_r2_combinable_p)(r1,r2) )
	{
	  combinable = TRUE;
	  l_res = gen_nconc((*r1_r2_binary_op)(r1,r2), l_res);
	  
	  /* gen_remove(&l2, EFFECT(CAR(lr2))); */
	  if (prec_lr2 != NIL)
	    CDR(prec_lr2) = CDR(lr2);
	  else
	    l2 = CDR(lr2);
	  
	  free(lr2); lr2 = NIL;
	  /* */
	  free_effect(r1); r1=effect_undefined; 
	  free_effect(r2); r2=effect_undefined;
	}
	else
	{
	  prec_lr2 = lr2;
	  lr2 = CDR(lr2);
	}
      }
      
      ifdebug(9)
	{
	  pips_debug(9, "intermediate effects 1:\n");
	  (*effects_prettyprint_func)(l_res);
	}
      
      if(!combinable)
      {
	/* r1 belongs to the remnants of l1 : it is combinable 
	 * with no effects of l2 */
	if ( (*r1_r2_combinable_p)(r1,effect_undefined) ) 
	  l_res = gen_nconc((*r1_unary_op)(r1), l_res);
      }
    },
	l1);
    
    ifdebug(9)
      {
	pips_debug(9, "intermediate effects 2:\n");
	(*effects_prettyprint_func)(l_res);
      }
    
    /* we must then deal with the remnants of l2 */
    MAP(EFFECT, r2,
    {  
      if ( (*r1_r2_combinable_p)(effect_undefined,r2) ) 
	l_res = gen_nconc((*r2_unary_op)(r2), l_res);
    },
	l2);
        
    ifdebug(1)
      {
	pips_debug(1, "final effects:\n");
	(*effects_prettyprint_func)(l_res);
      }
    
    /* no memory leaks: l1 and l2 won't be used anymore */
    gen_free_list(l1);
    gen_free_list(l2);
    
    debug_off();
    
    return l_res;
}

list 
proper_to_summary_effects(list l_effects)
{
    return proper_effects_combine(l_effects, FALSE);
}


/* list proper_effects_contract(list l_effects)
 * input    : a list of proper effects
 * output   : a list of proper effects in which there is no two identical 
 *            scalar effects. 
 * modifies : the input list. 
 * comment  : This is used to reduce the number of dependence tests.
 */

list 
proper_effects_contract(list l_effects)
{
    return(proper_effects_combine(l_effects, TRUE));
}


/* list proper_effects_combine(list l_effects, bool scalars_only_p)
 * input    : a list of proper effects, and a boolean to know on which
 *            elements to perform the combination.
 * output   : a list of effects, in which the selected elements have been 
 *            merged.
 * modifies : the input list.
 * comment  : the algorithm is in O(n) (was in (n^2)/2)
 * 
 * we need "entity/action" -> consp to check for the
 * condition in the second loop directly.
 * or to simplify the hash management, two entity -> consp?
 * a generic multi key combination hash would help.
 * the list is modified IN PLACE, storing on the first effect encountered...
 */
list 
proper_effects_combine(list l_effects, bool scalars_only_p)
{
  list cur, pred = NIL;
  /* entity name -> consp in effect list. */
  hash_table all_read_effects, all_write_effects;
  
  ifdebug(6) {
    pips_debug(6, "proper effects: \n");
    (*effects_prettyprint_func)(l_effects);	
  }
  
  all_read_effects = hash_table_make(hash_string, 10);
  all_write_effects = hash_table_make(hash_string, 10);

  cur = l_effects;
  /* scan the list of effects... the list is modified in place */
  while(!ENDP(cur))
  {
    effect current = EFFECT(CAR(cur));
    string n;
    tag a;
    bool may_combine, do_combine = FALSE;
    list do_combine_item = NIL;
    list next = CDR(cur); /* now, as 'cur' may be removed... */

    current = (*proper_to_summary_effect_func)(current);
    n = entity_name(effect_entity(current));
    a = effect_action_tag(current);

    /* may/do we have to combine ? */
    /* ??? FC this should be no big deal... anyway :
     * in the previous implementation, 'current' was not yet
     * passed thru proper_to_summary_effect_func when tested...
     */
    may_combine = !scalars_only_p || effect_scalar_p(current);

    if (may_combine)
    {
      /* did we see it? */
      switch (a) {
      case is_action_write:
	if (hash_defined_p(all_write_effects, n))
	{
	  do_combine = TRUE;
	  do_combine_item = hash_get(all_write_effects, n); 
	}
	break;
      case is_action_read:
	if (hash_defined_p(all_read_effects, n))
	{
	  do_combine = TRUE;
	  do_combine_item = hash_get(all_read_effects, n);
	}
	break;
      default: pips_internal_error("unexpected action tag %d", a);
      }
    }

    if (do_combine)
    {
      /* YES, me must combine */

      effect base = EFFECT(CAR(do_combine_item));
      /* compute their union */
      effect combined = (*effect_union_op)(base, current);
	
      /* free the original effects: no memory leak */
      free_effect(base);
      free_effect(current);
	
      /* replace the base effect by the new effect */
      EFFECT(CAR(do_combine_item)) = combined;
	
      /* remove the current list element from the global list */
      /* pred!=NIL as on the first items hash's are empty */
      CDR(pred) = next; /* pred is not changed! */
      free(cur);
    }
    else
    {
      /* NO, just store if needed... */
      EFFECT(CAR(cur)) = current;
      if (may_combine)
      {
	/* if we do not combine. ONLY IF we test, we put... */
	switch (a) {
	case is_action_write:
	  hash_put(all_write_effects, n, cur);
	  break;
	case is_action_read:
	  hash_put(all_read_effects, n, cur);
	  break;
	default: pips_internal_error("unexpected action tag %d", a);
	}
      }
      pred = cur;
    }

    cur = next;
  }
  
  ifdebug(6){
    pips_debug(6, "summary effects: \n"); 
    (*effects_prettyprint_func)(l_effects);	
  }

  hash_table_free(all_write_effects);
  hash_table_free(all_read_effects);

  return l_effects;
}



/******************************************************* BOOL(EAN) FUNCTIONS */

/* bool combinable_effects_p(effect eff1, eff2)
 * input    : two effects
 * output   : TRUE if eff1 and eff2 affect the same entity, and, if they
 *            have the same action on it, FALSE otherwise.
 * modifies : nothing.
 */
bool combinable_effects_p(effect eff1, effect eff2)
{
    bool same_var, same_act;

    if (effect_undefined_p(eff1) || effect_undefined_p(eff2))
	return(TRUE);

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    same_act = (effect_action_tag(eff1) == effect_action_tag(eff2));

    return(same_var && same_act);
}

bool effects_same_action_p(effect eff1, effect eff2)
{
    bool same_var, same_act;

    if (effect_undefined_p(eff1) || effect_undefined_p(eff2))
	return(TRUE);

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    same_act = (effect_action_tag(eff1) == effect_action_tag(eff2));

    return(same_var && same_act);
}

bool effects_same_variable_p(effect eff1, effect eff2)
{
    bool same_var = (effect_entity(eff1) == effect_entity(eff2));
    return(same_var);
}


bool r_r_combinable_p(effect eff1, effect eff2)
{
    bool same_var, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_read_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_read_p(eff1));

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    act_combinable = (effect_read_p(eff1) && effect_read_p(eff2));

    return(same_var && act_combinable);
}

bool w_w_combinable_p(effect eff1, effect eff2)
{
    bool same_var, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_write_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_write_p(eff1));

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    act_combinable = (effect_write_p(eff1) && effect_write_p(eff2));

    return(same_var && act_combinable);
}

bool r_w_combinable_p(effect eff1, effect eff2)
{
    bool same_var, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_write_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_read_p(eff1));

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    act_combinable = (effect_read_p(eff1) && effect_write_p(eff2));

    return(same_var && act_combinable);
}

bool w_r_combinable_p(effect eff1, effect eff2)
{
    bool same_var, act_combinable;

    if (effect_undefined_p(eff1))
	return(effect_read_p(eff2));

    if (effect_undefined_p(eff2))
	return(effect_write_p(eff1));

    same_var = (effect_entity(eff1) == effect_entity(eff2));
    act_combinable = (effect_write_p(eff1) && effect_read_p(eff2));

    return(same_var && act_combinable);
}

/***********************************************************************/
/* UNDEFINED BINARY OPERATOR                                           */
/***********************************************************************/

list
effects_undefined_binary_operator(list l1, list l2,
				  bool (*effects_combinable_p)(effect, effect))
{
  pips_assert("unused arguments", l1==l1 && l2==l2 &&
	      effects_combinable_p==effects_combinable_p);
  return list_undefined;
}


/***********************************************************************/
/* SOME BINARY OPERATORS which do not depend on the representation     */
/***********************************************************************/

/* list effect_entities_intersection(effect eff1, effect eff2)
 * input    : two effects
 * output   : a mere copy of the first effect.
 * modifies : nothing.
 * comment  : We assume that both effects concern the same entity.
 */
static list 
effect_entities_intersection(effect eff1, effect eff2)
{
  pips_assert("unused argument", eff2==eff2);
  return CONS(EFFECT, (*effect_dup_func)(eff1), NIL);
}

/* list effects_entities_intersection(list l1, list l2, 
                           bool (*intersection_combinable_p)(effect, effect))
 * input    : two lists of effects.
 * output   : a list of effects containing all the effects of l1 that have
 *            a corresponding effect (i.e. same entity) in l2.
 * modifies : l1 and l2.
 * comment  :	
 */
list
effects_entities_intersection(list l1, list l2,
			      bool (*intersection_combinable_p)(effect, effect))
{
    list l_res = NIL;

    pips_debug(3, "begin\n");
    l_res = list_of_effects_generic_binary_op(l1, l2,
					   intersection_combinable_p,
					   effect_entities_intersection,
					   effect_to_nil_list,
					   effect_to_nil_list);
    pips_debug(3, "end\n");

    return l_res;
}


/* list effects_entities_inf_difference(list l1, l2)
 * input    : two lists of effects
 * output   : a list of effects, such that: if there is a effect R concerning
 *            entity A in l1 and in l2, then R is removed from the result;
 *            if there is a effect R concerning array A in l1, but not in l2,
 *            then it is kept in l1, and in the result.
 * modifies : the effects of l2 may be freed.
 * comment  : we keep the effects of l1 that are not combinable with those
 *            of l2, but we don't keep the effects of l2 that are not 
 *            combinable with those of l_reg1.	
 */
list
effects_entities_inf_difference(
    list l1, 
    list l2,
    bool (*difference_combinable_p)(effect, effect))
{
    list l_res = NIL;

    debug(3, "RegionsEntitiesInfDifference", "begin\n");
    l_res = list_of_effects_generic_binary_op(l1, l2,
					   difference_combinable_p,
					   effects_to_nil_list,
					   effect_to_list,
					   effect_to_nil_list);
    debug(3, "RegionsEntitiesInfDifference", "end\n");

    return l_res;
}

/* that is all
 */
