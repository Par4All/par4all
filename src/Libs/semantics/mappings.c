 /* mappings package 
  *
  * Establish mappings between integer scalar variable entities and
  * value entities for a given module
  *
  * Handle equivalences
  *
  * Cannot handle more than one module at a time: no recursivity on
  * modules or chaos will occur
  *
  * See package value for more information on ri independent routines
  *
  * Francois Irigoin, 20 April 1990
  */

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"

#include "misc.h"

#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

#include "transformer.h"

#include "semantics.h"


/* EQUIVALENCE */

static Pcontrainte equivalence_equalities = CONTRAINTE_UNDEFINED;

static void 
reset_equivalence_equalities()
{
    if(equivalence_equalities != CONTRAINTE_UNDEFINED) {
	equivalence_equalities = contraintes_free(equivalence_equalities);
    }
}

transformer 
tf_equivalence_equalities_add(tf)
transformer tf;
{
    /* I need here a contraintes_dup() that is not yet available
       in Linear and I cannot change Linear just before the DRET meeting;
       I've got to modify transformer_equalities_add() and to give it
       a behavior different from transformer_equality_add() */
    tf = transformer_equalities_add(tf, equivalence_equalities);
    return tf;
}

static void 
add_equivalence_equality(e, eq)
entity e;
entity eq;
{
    /* assumes e and eq are different */
    Pvecteur v = vect_new((Variable) e, 1);
    Pcontrainte c;

    vect_add_elem(&v, (Variable) eq, -1);
    c = contrainte_make(v);
    /* Strange: there is no macro to chain a contrainte on a list of
       contrainte */
    c->succ = equivalence_equalities;
    equivalence_equalities = c;
}

void 
add_equivalenced_values(e, eq, readonly)
entity e;
entity eq;
bool readonly;
{
    /* e and eq are assumed to be integer scalar variables */
    /* eq will be seen as e, as far as values are concerned,
       but for printing */
    /* new equalities e#new == eq#new and e#old == eq#old
       have to be added to the preconditions just before they
       are stored; since eq should never be written no eq#old
       should appear; thus the first equation is enough */

    /* apparently the parser sees all variables as conflicting
       with themselves */
    if(e!=eq) {
	add_synonym_values(e, eq, readonly);
	/* add the equivalence equations */
	add_equivalence_equality(e, eq);
    }
}

/* ???? */

/* void add_interprocedural_value_entities 
 */
static void 
add_interprocedural_value_entities(e)
entity e;
{
    debug(8,"add_interprocedural_value_entities","for %s\n", 
	  entity_name(e));
    if(!entity_has_values_p(e)) {
	entity a = entity_undefined;
	if((a=value_alias(e))==entity_undefined){
	    add_new_value(e);
	    add_old_value(e);
	    add_intermediate_value(e);
	    add_or_kill_equivalenced_variables(e, FALSE);
	}
	else {
	    add_new_alias_value(e,a);
	    add_old_alias_value(e,a);
	    add_intermediate_alias_value(e,a);
	}
    }
}
static void 
add_interprocedural_new_value_entity(e)
entity e;
{
    debug(8,"add_interprocedural_new_value_entities","for %s\n", 
	  entity_name(e));
    if(!entity_has_values_p(e)) {
	entity a = entity_undefined;
	if((a=value_alias(e))==entity_undefined){
	    add_new_value(e);
	    /* CA: information on aliasing variables erased*/
	    add_or_kill_equivalenced_variables(e,TRUE);
	}
	else {
	    add_new_alias_value(e,a);
	}
    }
}

static void 
add_intraprocedural_value_entities_unconditionally(entity e)
{
    debug(8,"add_interprocedural__value_entities_unconditionally",
	  "for %s\n", entity_name(e));
	add_new_value(e);
	add_local_old_value(e);
	add_local_intermediate_value(e);
	add_or_kill_equivalenced_variables(e, FALSE);
}

/* void add_intraprocedural_value_entities(entity e)
 */
static void 
add_intraprocedural_value_entities(entity e)
{ 
    debug(8,"add_interprocedural__value_entities",
	  "for %s\n", entity_name(e));
    if(!entity_has_values_p(e)) {
	add_intraprocedural_value_entities_unconditionally(e);
    }
}

/*  */

void 
add_or_kill_equivalenced_variables(e, readonly)
entity e;
bool readonly;
{
    /* look for equivalenced variables; forget dynamic aliasing
       between formal parameters */
    storage s;
 
    debug(8,"add_or_kill_equivalenced_variables",
	  "for %s\n", entity_name(e));
    s = entity_storage(e);
    if(storage_ram_p(s)) {
	/* handle intraprocedural aliasing */
	list local_shared = ram_shared(storage_ram(s));
	/* handle interprocedural aliasing: this does not seem to be the right place
	 * because too many synonyms, not visible from the current procedure, are
	 * introduced */
	/* list global_shared = NIL; */
	bool array_equivalenced = FALSE;
	entity sec = ram_section(storage_ram(s));
	type t = entity_type(sec);

	pips_assert("add_or_kill_equivalenced_variables", type_area_p(t));

	/* global_shared = area_layout(type_area(t)); */

	/* Is e intra or interprocedurally equivalenced/aliased with an array or a
	 * non-integer variable?
	 */
	MAPL(ce, {
	    entity eq = ENTITY(CAR(ce));
	    /* this approximate test by Pierre Jouvelot should be
	       replaced by an exact test but it does not really matter;
	       an exact test could only be useful in presence of arrays;
	       and in presence of arrays we do nothing here */
	    if(entity_conflict_p(e, eq) && !integer_scalar_entity_p(eq)) {
		pips_user_warning("Values for variable %s are not analyzed because "
				  "%s is aliased with non scalar integer variable %s\n",
				  entity_name(e), entity_name(e), entity_name(eq));
		array_equivalenced = TRUE;
		break;
	    }
	},
	     local_shared);

	/* if it's not, go ahead */
	if(!array_equivalenced) {

	    /* If it is intra-procedurally equivalenced, set the synonyms as
	     * read-only variables
	     */
	    MAPL(ce, {
		entity eq = ENTITY(CAR(ce));
		if(entity_conflict_p(e, eq)) {
		    if(integer_scalar_entity_p(eq)) {
			/* if eq is an integer scalar variable it does not
			   only have a destructive effect */
			add_equivalenced_values(e, eq, readonly);
		    }
		}
	    },
		 local_shared); 

	    /* If it is inter-procedurally aliased, set the synonyms as
	     * regular variables
	     */
	    /* FI: This is damaging because too many equivalences are introduced;
	     * synonyms that are not visible from the current procedure are added
	     * because they are visible from the main, i.e. the whole program
	     * is (assumed) analyzed.
	    MAPL(ce, {
		entity eq = ENTITY(CAR(ce));
		if(entity_conflict_p(e, eq) && !entity_is_argument_p(eq, local_shared)) {
		    if(integer_scalar_entity_p(eq)) {
			add_equivalenced_values(e, eq, FALSE);
		    }
		}
	    },
		 global_shared); 
		 */
	}
	else {
	    /* Variable e is equivalenced with an array or a non-integer
	     * variable and cannot be analyzed; it must be removed from
	     * the hash tables.
	     */
	    remove_entity_values(e, readonly);
	}
    }
    else if(storage_return_p(s))
	/* semantics analysis should be performed on this kind of variable
	   but it has probably been eliminated earlier; no equivalence
	   possible anyway! */
	user_warning("add_or_kill_equivalenced_variables","storage return\n");
    else if(storage_formal_p(s))
	/* to be dealt with later if we assume non-standard dynamic
	   aliasing between formal parameters */
	;
    else
	pips_error("add_or_kill_equivalenced_variables",
		   "unproper storage = %d\n", storage_tag(s));
}

static void 
allocate_module_value_mappings(entity m)
{
    /* this routine tries to estimate the sizes of the hash tables,
       although the hashtable package has enlarging capability;
       its usefulness is limited... but keep at least hash table
       allocations! */
    
    list module_intra_effects = load_module_intraprocedural_effects(m);
    int old_value_number = 0;
    int intermediate_value_number = 0;
    int new_value_number = 0;

    /* count interprocedural effects on scalar integer variables
       before allocating hash tables; too many entries might be
       expected because the same variable could appear many times,
       at least twice, once in a read effect and once in a write 
       effect; entries for arrays equivalenced with scalar variables
       are ignored; some slack is added before allocating the hash
       tables; module_inter_effects are (should be) included into
       module_intra_effects */
    MAP(EFFECT, ef,
    {
	entity e = reference_variable(effect_reference(ef));
	action a = effect_action(ef);
	if(integer_scalar_entity_p(e))
	    new_value_number++;
	if(action_write_p(a))
	    old_value_number++;
    },
	module_intra_effects);

    /* add 50 % slack for underestimation (some more slack will be added
       by the hash package */
    new_value_number *= 3; new_value_number /= 2;
    old_value_number *= 3; old_value_number /= 2;
    /* the hash package does not like very small sizes */
    new_value_number = MAX(10,new_value_number);
    old_value_number = MAX(10,old_value_number);
    /* overapproximate intermediate value number */
    intermediate_value_number = old_value_number;

    pips_debug(8, "old_value_number = %d\n", old_value_number);
    pips_debug(8, "new_value_number = %d\n", new_value_number);

    /* allocate hash tables */
    allocate_value_mappings(new_value_number, old_value_number,
			    intermediate_value_number);
}

/* void module_to_value_mappings(entity m): build hash tables between
 * variables and values (old, new and intermediate), and between values
 * and names for module m, as well as equivalence equalities
 *
 * NW:
 * before calling "module_to_value_mappings"
 * to set up the hash table to translate value into value names
 * for module with name (string) module_name
 * do:
 *
 * set_current_module_entity( local_name_to_top_level_entity(module_name) );
 *
 * (the following call is only necessary if a variable of type entity
 * such as "module" is not already set)
 * module = get_current_module_entity();
 *
 * set_current_module_statement( (statement)
 * 			      db_get_memory_resource(DBR_CODE,
 * 						     module_name,
 * 						     TRUE) );
 * set_cumulated_rw_effects((statement_effects)
 * 			 db_get_memory_resource(DBR_CUMULATED_EFFECTS,
 * 						module_name,
 * 						TRUE));
 * 
 * (that's it, but we musn't forget to reset everything
 * after the call to "module_to_value_mappings", as below)
 *
 * reset_current_module_statement();
 * reset_cumulated_rw_effects();
 * reset_current_module_entity(); 
 * free_value_mappings();
 */
void 
module_to_value_mappings(m)
entity m;
{
    cons * module_inter_effects;
    cons * module_intra_effects;

    pips_debug(8,"begin\n");
    pips_debug(8,"module = %s\n", module_local_name(m));

    pips_assert("m is a module", entity_module_p(m));

    /* free_value_mappings(); */

    allocate_module_value_mappings(m);

    /* reset local intermediate value counter for 
       make_local_intermediate_value_entity and make_local_old_value_entity */
    reset_value_counters();
    reset_equivalence_equalities();

    /* module_inter_effects = code_effects(value_code(entity_initial(m))); */
    module_inter_effects = load_summary_effects(m);

    /* look for interprocedural write effects on scalar integer variables
       and generate proper entries into hash tables */
    MAPL(cef,
     {entity e = 
	  reference_variable(effect_reference(EFFECT(CAR(cef))));
	  action a = effect_action(EFFECT(CAR(cef)));
	  if(integer_scalar_entity_p(e) && action_write_p(a)) 
	      add_interprocedural_value_entities(e);
      },
	 module_inter_effects);

    /* look for interprocedural read effects on scalar integer variables
       and generate proper entries into hash tables */
    MAPL(cef,
     {entity e = 
	  reference_variable(effect_reference(EFFECT(CAR(cef))));
	  action a = effect_action(EFFECT(CAR(cef)));
	  if(integer_scalar_entity_p(e) && action_read_p(a)) 
	      add_interprocedural_new_value_entity(e);
      },
	 module_inter_effects);

    /* module_intra_effects = 
     * load_statement_cumulated_effects(code_statement(value_code(entity_initial(m))));
     */

    module_intra_effects = 
	load_module_intraprocedural_effects(m);

    /* look for intraprocedural write effects on scalar integer variables
       and generate proper entries into hash tables */
    MAP(EFFECT, ef,
    {
	entity e = reference_variable(effect_reference(ef));
	action a = effect_action(ef);
	if(integer_scalar_entity_p(e) && action_write_p(a)) 
	    if(storage_return_p(entity_storage(e))) {
		add_interprocedural_value_entities(e);
	    }
	    else {
		add_intraprocedural_value_entities(e);
	    }
    },
	 module_intra_effects);
    
    /* look for intraprocedural read effects on scalar integer variables
       and generate proper entry into value name hash table if it has
       not been entered before; interprocedural read effects are implicitly
       dealed with since they are included; 
       most entities are likely to have been encountered before; however
       in parameters and uninitialized variables have to be dealt with */
    MAP(EFFECT, ef,
     {
	 entity e = reference_variable(effect_reference(ef));
	 if(integer_scalar_entity_p(e) && !entity_has_values_p(e)) {
	      /* FI: although it may only be read within this procedure,
	       * e might be written in another one thru a COMMON;
	       * this write is not visible from OUT, but only from a caller
	       * of out; because we have only a local intraprocedural or a 
	       * global interprocedural view of aliasing, we have to create 
	       * useless values:-(
	       *
	       * add_new_value(e);
	       *
	       * Note: this makes the control structure of this procedure 
	       * obsolete!
	       */
	     /* This call is useless because it only is effective if
	      * entity_has_values_p() is true:
	      * add_intraprocedural_value_entities(e);
	      */
	      add_intraprocedural_value_entities_unconditionally(e);
	      /* A stronger call to the same subroutine is included in
	       * the previous call:
	       * add_or_kill_equivalenced_variables(e, TRUE);
	       */
	  }},
	 module_intra_effects);

    /* scan declarations to make sure that private variables are
       taken into account; assume a read and write effects on these
       variables, although they may not even be used.

       Only intraprocedural variables can be privatized (1 Aug. 92) */
    MAPL(ce,
     {entity e = ENTITY(CAR(ce));
	  if(integer_scalar_entity_p(e) && !entity_has_values_p(e)) {
	      add_intraprocedural_value_entities(e);
	  }}, code_declarations(value_code(entity_initial(m))));

    /* for debug, print hash tables */
    ifdebug(8) {
	
	(void) fprintf(stderr,"[module_to_value_mappings] hash tables\n");
	print_value_mappings();
	test_mapping_entry_consistency();
	
    }
    pips_debug(8,"end\n");
}

/* transform a vector based on variable entities into a vector based
 * on new value entities; does nothing most of the time; does a little
 * in the presence of equivalenced variables
 *
 * Ugly because it has a hidden side effect on v and because it's
 * implementation dependent on type Pvecteur
 */
bool 
value_mappings_compatible_vector_p(v)
Pvecteur v;
{
    for(;!VECTEUR_NUL_P(v); v = v->succ) {
	if(vecteur_var(v) != TCST) {
	    if(entity_has_values_p((entity) vecteur_var(v))) {
	    entity new_v = entity_to_new_value((entity) vecteur_var(v));
	
	    if(new_v != entity_undefined)
		vecteur_var(v) = (Variable) new_v;
	    else
		return FALSE;
	    }
	    else {
		return FALSE;
	    }
	}
    }
    return TRUE;
}
