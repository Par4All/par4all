/* package scalarization
 *
 * $Id$
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "pipsmake.h"
#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "transformations.h"


////////////////////

/*
 * WARNING ! This function introduces side-effects on g
 */

Psysteme sc_add_offset_variables(Psysteme g, Pbase b, Pbase db)
{
  Pbase gb = sc_base(g);

  // Check validity conditions: b must appear in g's basis but db must not, and dim(b) == dim(db)
  if (base_included_p(b, gb) && !base_included_p(db, gb) && base_dimension(b)==base_dimension(db)) {

    Pcontrainte eqs   = sc_egalites(g);
    Pcontrainte ineqs = sc_inegalites(g);
    
    Pcontrainte c = CONTRAINTE_UNDEFINED;

    // Update g's basis
    Pbase tb = sc_base(g);
    sc_base(g) = base_union(tb, db);
    sc_dimension(g) += base_dimension(db);
    base_rm(tb);

    for( c = eqs ; !CONTRAINTE_UNDEFINED_P(c) ; c= contrainte_succ(c) ) {
      Pbase cb  = BASE_NULLE;
      Pbase cdb = BASE_NULLE;
      for ( cb=b,cdb=db ; !BASE_NULLE_P(cb) ; cb = vecteur_succ(cb), cdb = vecteur_succ(cdb)) {
	Variable v     = vecteur_var(cb);
 	Variable dv    = vecteur_var(cdb);
	Value    coeff = vect_coeff(v, contrainte_vecteur(c));	
	vect_add_elem(&(contrainte_vecteur(c)), dv, coeff);
      }
    }

    for( c = ineqs ; !CONTRAINTE_UNDEFINED_P(c) ; c= contrainte_succ(c) ) {
      Pbase cb  = BASE_NULLE;
      Pbase cdb = BASE_NULLE;
      for ( cb=b,cdb=db ; !BASE_NULLE_P(cb) ; cb = vecteur_succ(cb), cdb = vecteur_succ(cdb)) {
	Variable v     = vecteur_var(cb);
	Variable dv    = vecteur_var(cdb);
	Value    coeff = vect_coeff(v, contrainte_vecteur(c));	
	vect_add_elem(&(contrainte_vecteur(c)), dv, coeff);
      }
    }
  }

  return g;
}


/* 
 * This function checks that graph g is a function graph from domain d
 * to range r.
 *
 * Return value : TRUE if the graph is certainly
 * functional, FALSE if it *might* be not functional.
 *
 * NOTE: parameter dr could be derived from r within this function,
 * but we do not know how to create generic variables in Linear. The
 * initial implementation used a PIPS function to generate the new
 * variables.
 */

bool sc_functional_graph_p(Psysteme g, Pbase d, Pbase r, Pbase dr)
{
  bool functional_p = TRUE;

  Psysteme g1, g2, g3, g4;

  // Check validity conditions: d and r should be included in g's basis.
  Pbase gb = sc_base(g);
  if (!(base_included_p(d, gb) && base_included_p(r, gb))) {

    // Illegal arguments
    functional_p = FALSE; //TODO
  }
  else {

    Pbase cr = BASE_UNDEFINED;

    // Create two copies of g
    g1 = sc_copy(g);
    g2 = sc_copy(g);
    
    // Substitute r by r + dr in g2 (indexed by dimension)
    g2 = sc_add_offset_variables(g2, r, dr);

    // Merge g1 and g2 into a unique system g3.
    g3 = sc_intersection(SC_EMPTY, g1, g2);
    g3 = sc_elim_redond(g3); // remove redundencies
    
    // Project g3 on dr space -> g4. If projection fails, return FALSE.
    Pbase dr4 = BASE_NULLE;
    // dr4 := list of variables of g3's basis which are not in dr
    for ( cr = sc_base(g3) ; !BASE_UNDEFINED_P(cr) ; cr = vecteur_succ(cr) ) {
      Variable vr = vecteur_var(cr);
      if (!base_find_variable(dr, vr))
	dr4 = base_add_variable(dr4, vr);
    }
    g4 = sc_copy(g3);
    sc_projection_along_variables_ofl_ctrl(&g4, dr4, OFL_CTRL);
    base_rm(dr4);
    ifdebug(1) {
      fprintf(stderr, "g4 =\n");
      sc_print(g4, (get_variable_name_t)entity_local_name);
    }
    if (SC_EMPTY_P(g4)) {
      functional_p = FALSE;
    }
    else {
    
      // Check that all r_b_i variables are null, using sc_minmax_of_variables()    
      for ( cr = dr ; !BASE_UNDEFINED_P(cr) ; cr = vecteur_succ(cr) ) {
	Psysteme g4b = sc_copy(g4);
	Variable dv  = vecteur_var(cr);
	Value pmin, pmax;
	bool feasible_p = sc_minmax_of_variable(g4b,dv, &pmin, &pmax);
	if (!(feasible_p && value_eq(VALUE_ZERO, pmin) && value_eq(VALUE_ZERO, pmax))) {
	  functional_p = FALSE;
	  break;
	}
      }
    }
  }
  sc_rm(g1);
  sc_rm(g2);
  sc_rm(g3);
  sc_rm(g4);

  return functional_p;
}


bool sc_totally_functional_graph_p( Psysteme g, // function graph
				    Pbase d,    // domain's basis
				    Psysteme D, // membership predicate for functional domain
				    Pbase r,    // range's predicate
				    Pbase dr    // difference variable
				    )
{
  bool totally_functional_p = FALSE;
  
  if (sc_functional_graph_p(g, d, r, dr)) {

    // Check args coherence : d should be included in D's basis.
    if (base_included_p(d, sc_base(D))) {  

      // Project g on d along r.
      Psysteme g1 = sc_copy(g);
      sc_projection_along_variables_ofl_ctrl(&g1, r, OFL_CTRL);
      
      // By definition of a total function, D must be included in g1.
      totally_functional_p = sc_inclusion_p_ofl_ctrl(D, g1, OFL_CTRL);
    }
  }
  return totally_functional_p;
}


// We need a Pbase to accumulate loop indices during loop traversal
static Pbase loop_indices_b = BASE_NULLE;

// Needed for callback function reference_substitute
static entity scalarized_array = entity_undefined;
static entity scalarized_replacement_variable = entity_undefined;

// gen_recurse callback function for statement_substitute_scalarized_array_references
static bool reference_substitute(reference r) {
  entity v = reference_variable(r);
  if (v == scalarized_array) {
    reference_variable(r) = scalarized_replacement_variable;
    reference_indices(r) = NIL; // TODO: add missing gen_full_free_list(reference_indices(r))
  }
  return TRUE;
}


static void statement_substitute_scalarized_array_references(statement st, entity a, entity s)
{
  //TODO: create context ansd use gen_multi_recurse_with_context
  scalarized_array = a;
  scalarized_replacement_variable = s;
  gen_recurse (st, reference_domain, reference_substitute, gen_null);
  scalarized_array = entity_undefined;
  scalarized_replacement_variable = entity_undefined;  
}


static void * car_effect_to_variable(gen_chunk car) {
  return effect_variable(EFFECT(car)); // type 'entity'
}


Pbase make_phi_base(int phi_min, int phi_max)
{
  Pbase phi_b = BASE_NULLE;
  int i;
  for( i=phi_min; i<=phi_max; i++ ) 
    phi_b = base_add_variable(phi_b, (Variable) make_phi_entity(i));
  return(phi_b);
}


// gen_recurse callback on entering statement. If it's a loop, process it.
static bool loop_in(statement ls)
{
  if (statement_loop_p(ls)) {

    loop l = statement_loop(ls);

    entity i    = loop_index(l);
    statement s = loop_body(l);

    transformer prec = load_statement_precondition(s);
    Psysteme D = predicate_system(transformer_relation(transformer_range(prec)));

    effects ie  = load_in_effects(s);
    effects oe  = load_out_effects(s);
    effects pe  = load_private_effects(s);
    effects coe = load_copy_out_effects(s);

    list irl  = effects_effects(ie);
    list orl  = effects_effects(oe);
    list prl  = effects_effects(pe);
    list corl = effects_effects(coe);

    // Accumulate new index in "domain" basis
    loop_indices_b = base_add_variable(loop_indices_b, (Variable) i);

    ifdebug(1) {
      fprintf(stderr, "Entering level-%d loop, index=%s\n", base_dimension(loop_indices_b), entity_name(i));
      fprintf(stderr, "PRIVATIZED regions:");
      print_regions(prl);
      fprintf(stderr, "COPY OUT regions:");
      print_regions(corl);
    }

    // Now we determine which private effects are not copied out. COPY IN effects are not implemented yet
    // LD 2009/04/10.
    FOREACH (EFFECT, pr, prl) {
      entity pv  = effect_variable(pr);
      entity iv  = (entity) gen_find(pv,  irl, (bool (*)())gen_eq, car_effect_to_variable);
      entity ov  = (entity) gen_find(pv,  orl, (bool (*)())gen_eq, car_effect_to_variable);
      entity cov = (entity) gen_find(pv, corl, (bool (*)())gen_eq, car_effect_to_variable);
      descriptor d = effect_descriptor(pr);
    
      if ( descriptor_convex_p(d)  &&
	   entity_undefined_p(cov) && // pv can be scalarized because it's not copied out
	   // No test on not-yet-implemented COPY IN
	   entity_undefined_p(iv)  && // pv can be scalarized because it's not in an in region
	   entity_undefined_p(ov)     // pv can be scalarized because it's not in an out region
	   ) {
	
	Psysteme sc = descriptor_convex(d);
	int nd = type_depth(entity_type(pv));

	//if (!entity_scalar_p(pv))
	if (nd > 0) {

	  Pbase phi_b = make_phi_base(1, nd);
	  Pbase d_phi_b = BASE_NULLE;
	  Pbase cr = BASE_NULLE;

	  ifdebug(1) {
	    fprintf(stderr, "LOOP_IN: Value of sc:\n");
	    sc_print(sc, (get_variable_name_t)entity_user_name);
	  }
	
	  // Build base dr using make_local_temporary_integer_value_entity(void)
	  for ( cr = phi_b ; !BASE_UNDEFINED_P(cr) ; cr = vecteur_succ(cr) ) {
	    entity e_d_phi_b = make_local_temporary_integer_value_entity();
	    d_phi_b = base_add_variable(d_phi_b, (Variable) e_d_phi_b);
	  }

	  if (sc_totally_functional_graph_p(sc, loop_indices_b, D, phi_b, d_phi_b)) {
	    // Create new temp var of same type as pv
	    type pvt      = ultimate_type(entity_type(pv)); // ultime_type "un-hides" typedefs
	    variable pvtv = type_variable(pvt);
	    basic pvb     = variable_basic(pvtv);
	    basic svb     = copy_basic(pvb);      
	  
	    // Create a reference to this new variable and add declaration to module
	    entity sv = make_new_scalar_variable_with_prefix("__ld__", get_current_module_entity(), svb);
	    AddEntityToCurrentModule(sv);
	  
	    // Substitute all references to pv with references to new variable
	    statement_substitute_scalarized_array_references(s, pv, sv);

	  }
	  base_rm(phi_b);
	  base_rm(d_phi_b);
	  reset_temporary_value_counter();
	}
      }
    }
  }
  return TRUE;
}


// gen_recurse callback on exiting loop
static void loop_out(statement s)
{
  if (statement_loop_p(s)) {
    loop l = statement_loop(s);
    entity i = loop_index(l);
    ifdebug(1) {
      fprintf(stderr, "Exiting loop with index %s, size=%d\n", entity_name(i), base_dimension(loop_indices_b));
    }
    loop_indices_b = base_remove_variable(loop_indices_b, (Variable) i);

  }
}


bool scalarization (char * module_name)
{
    entity module;
    statement module_stat;

    set_current_module_entity(module_name_to_entity(module_name));
    module = get_current_module_entity();
 
    set_current_module_statement( (statement)
	     db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();

    //set_proper_rw_effects((statement_effects) 
    //db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
    set_cumulated_rw_effects((statement_effects) 
			     db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

    module_to_value_mappings(module);
  
    set_precondition_map((statement_mapping) 
			 db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE));
    set_in_effects((statement_effects)
		   db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE));
    set_out_effects((statement_effects)
		    db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE));
    set_private_effects((statement_effects)
			db_get_memory_resource(DBR_PRIVATIZED_REGIONS, module_name, TRUE));
    set_copy_out_effects((statement_effects)
			 db_get_memory_resource(DBR_COPY_OUT_REGIONS, module_name, TRUE));      

    debug_on("SCALARIZATION_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    // ifdebug(1) print_statement(module_stat);

    /* We now traverse our module's loops. */
    loop_indices_b = BASE_NULLE;
    gen_recurse(module_stat, statement_domain, loop_in, loop_out);

    pips_debug(1, "end\n");
    debug_off();

    /* Save modified code to database */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

    /* TODO: Cleanup after scalarization */
    pips_assert("Loop index Pbase is empty", BASE_NULLE_P(loop_indices_b));

    reset_current_module_entity();
    reset_current_module_statement();

    //reset_proper_rw_effects();
    reset_cumulated_rw_effects();

    reset_precondition_map();
    reset_in_effects();
    reset_out_effects();
    reset_private_effects();
    reset_copy_out_effects();
    
    free_value_mappings();

    /* Return value */
    bool good_result_p = TRUE;

    return (good_result_p);

}
