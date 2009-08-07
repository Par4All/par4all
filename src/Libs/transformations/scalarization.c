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
/* package scalarization
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
    /*
      ifdebug(1) {
      pips_debug(1, "g4 =\n");
      sc_print(g4, (get_variable_name_t)entity_local_name);
      }
    */
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
  // Cleanup
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


// To store already scalarized vars
static list scalarized_variables = list_undefined;

static bool loop_scalarization(loop l)
{
  entity i    = loop_index(l);
  statement s = loop_body(l);

  ifdebug(1) {
    pips_debug(1, "Statement:\n");
    print_statement(s);
  }

  transformer prec = load_statement_precondition(s);
  Psysteme D       = predicate_system(transformer_relation(transformer_range(prec)));

  effects ie   = load_in_effects(s);
  effects oe   = load_out_effects(s);
  effects pe   = load_private_effects(s);
  effects coe  = load_copy_out_effects(s);
  effects crwe = load_cumulated_rw_effects(s);

  list irl  = effects_effects(ie);
  list orl  = effects_effects(oe);
  list prl  = effects_effects(pe);
  list corl = effects_effects(coe);
  list crwl = effects_effects(crwe);

  ifdebug(1) {
    pips_debug(1, "Entering function...\n");
    pips_debug(1, "Entering level-%d loop, index=%s\n", base_dimension(loop_indices_b), entity_name(i));
    pips_debug(1, "OUT regions:\n");
    print_regions(orl);
    pips_debug(1, "PRIVATIZED regions:\n");
    print_regions(prl);
    pips_debug(1, "COPY OUT regions:\n");
    print_regions(corl);
    pips_debug(1, "CUMULATED RW regions:\n");
    print_regions(crwl);
  }

  // Now we determine which private effects are not copied out.
  FOREACH (EFFECT, pr, crwl) {
    entity pv  = effect_variable(pr);
    entity iv  = (entity) gen_find(pv, irl, (bool (*)())gen_eq, car_effect_to_variable);
    entity ov  = (entity) gen_find(pv, orl, (bool (*)())gen_eq, car_effect_to_variable);
    //entity cov = (entity) gen_find(pv, corl, (bool (*)())gen_eq, car_effect_to_variable);

    descriptor d = effect_descriptor(pr);    
    //action a = effect_action(pr);

    if ( /* action_write_p(a) && */
	 descriptor_convex_p(d) &&
	 !entity_is_argument_p(pv, scalarized_variables)
	 ) {
	
      Psysteme sc = descriptor_convex(d);
      int nd = type_depth(entity_type(pv));

      // Number of occurrences
      int no = count_references_to_variable(s, pv);

      bool read_pv    = effects_read_variable_p(crwl, pv);
      bool written_pv = effects_write_variable_p(crwl, pv);
      bool read_and_written_pv = read_pv && written_pv;


      //if (!entity_scalar_p(pv)) -- replaced by 'nd > 0'

      /* Profitability:

	 - no > 2: if the number of references if greater than 2, the
           copy-in and copy-out code overhead is assumed to be small
           enough to make scalarization profitable.
	   
	 - no > 1: if the number of references is 2, the copy-in *xor*
           the copy-out overhead meets the above criterion.

	 - else: if there is neither copy-in nor copy-out,
           privatization is always useful.

       */
      if (nd > 0
	  && (no > 2 
	      || (no > 1 && !read_and_written_pv)
	      || (entity_undefined_p(iv) && entity_undefined_p(ov))
	      )
	  ) {

	
	Pbase phi_b = make_phi_base(1, nd);
	Pbase d_phi_b = BASE_NULLE;
	Pbase cr = BASE_NULLE;

	/*
	  ifdebug(1) {
	  pips_debug(1, "Value of sc:\n");
	  sc_print(sc, (get_variable_name_t)entity_user_name);
	}
	*/
	
	// Build base dr using make_local_temporary_integer_value_entity(void)
	for ( cr = phi_b ; !BASE_UNDEFINED_P(cr) ; cr = vecteur_succ(cr) ) {
	  entity e_d_phi_b = make_local_temporary_integer_value_entity();
	  d_phi_b = base_add_variable(d_phi_b, (Variable) e_d_phi_b);
	}

	if (sc_totally_functional_graph_p(sc, loop_indices_b, D, phi_b, d_phi_b)) {
	  // Create new temp var of same type as pv
	  type pvt      = ultimate_type(entity_type(pv)); // ultimate_type "un-hides" typedefs
	  variable pvtv = type_variable(pvt);
	  basic pvb     = variable_basic(pvtv);
	  basic svb     = copy_basic(pvb);      
	  
	  //list el       = load_proper_rw_effects_list(s);

	  // Copy the a reference to pv, just in case we need it later
	  reference pvr = copy_reference(find_reference_to_variable(s, pv));

	  // Create a reference to this new variable and add declaration to module	
	  entity sv = make_new_scalar_variable_with_prefix("__ld__", get_current_module_entity(), svb);
	  AddEntityToCurrentModule(sv);
	  scalarized_variables = arguments_add_entity(scalarized_variables, pv);
	  
	  pips_user_warning("Creating variable %s for variable %s\n", entity_name(sv), entity_name(pv));	  
	  
	  // Substitute all references to pv with references to new variable	  
	  statement_substitute_scalarized_array_references(s, pv, sv);

	  //if (!entity_undefined_p(cov)) {
	  if (!entity_undefined_p(ov)) {
	    // Generate copy-out code
	    statement co_s = make_assign_statement(reference_to_expression(pvr), entity_to_expression(sv));
	    append_a_statement(s, co_s);
	  }
	  else {
	    //free_reference(pvr);
	  }

	  if (!entity_undefined_p(iv)) {
	    // Generate copy-in code
	    statement ci_s =
	      make_assign_statement(entity_to_expression(sv),
				    reference_to_expression(copy_reference(pvr)));
	    insert_a_statement(s, ci_s);
	  }
	  else {
	    //free_reference(pvr);
	  }

	}
	base_rm(phi_b);
	base_rm(d_phi_b);
	reset_temporary_value_counter();
      }
    }
  }
  return TRUE;
}


// gen_recurse callback on entering statement. If it's a loop, process it.
static bool statement_in(statement ls)
{
  bool result = TRUE;
  if (statement_loop_p(ls)) {
    loop l = statement_loop(ls);

    /* Insert a marker to keep track of the privatized variables
       inside loop l, lest they are privatized a second time in an inner
       loop.
    */
    entity i = loop_index(l);    
    scalarized_variables = arguments_add_entity(scalarized_variables, i);

    // Accumulate new index in "domain" basis
    loop_indices_b = base_add_variable(loop_indices_b, (Variable) i);

    result = loop_scalarization(l);
  }
  return result;
}


// gen_recurse callback on exiting loop
static void statement_out(statement s)
{
  if (statement_loop_p(s)) {
    loop l = statement_loop(s);
    entity i = loop_index(l);
    list nl = NIL;

    ifdebug(1) {
      pips_debug( 1, "Exiting loop with index %s, size=%d\n",
		  entity_name(i), base_dimension(loop_indices_b));
    }

    /* Remove variables privatized in the current look, so that
       successive loops don't interfere with each other.
    */
    
    for (list el=scalarized_variables; !ENDP(el); POP(el)) {
      entity e = ENTITY(CAR(el));
      if (e == i) {
	break;
      }
      else {
	nl = CONS(ENTITY, e, nl);
      }
    }
    gen_free_list(scalarized_variables);
    scalarized_variables = gen_nreverse(nl);

    //
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

    set_proper_rw_effects((statement_effects) 			  
			  db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));

    set_cumulated_rw_effects((statement_effects) 
			     db_get_memory_resource(DBR_REGIONS, module_name, TRUE));
    //    set_cumulated_rw_effects((statement_effects) 
    //			     db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

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

    /* We now traverse our module's statements. */
    loop_indices_b = BASE_NULLE;
    scalarized_variables = NIL;
    gen_recurse(module_stat, statement_domain, statement_in, statement_out);
    scalarized_variables = list_undefined;

    pips_debug(1, "end\n");
    debug_off();

    /* Save modified code to database */
	module_reorder(module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

    /* TODO: Cleanup after scalarization */
    pips_assert("Loop index Pbase is empty", BASE_NULLE_P(loop_indices_b));

    reset_current_module_entity();
    reset_current_module_statement();

    reset_proper_rw_effects();
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
