#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

/*static_control*/
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "ri.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
//typedef void * arc_label;
//typedef void * vertex_label;
#include "phrase.h"
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "paf-util.h"
#include "static_controlize.h"
#include "pocc-interface.h"
#include "preprocessor.h"
#include "effects-convex.h"
#include "callgraph.h" // For compute_callees()
#include "effects-generic.h" // For compute_callees()
#include "accel-util.h" // For outliner()
#include "c_syntax.h"
#include "syntax.h"
#include "contrainte.h"
#include "vecteur.h"
#include "semantics.h"
#include "regions_to_loops.h"

/* Filter the region list by removing irrelevant regions */
static void filter_regions(list* l) {
  list l_temp = gen_copy_seq(*l);
  descriptor d;
  reference r;
  list indices;
  FOREACH(effect, e, l_temp) {
    d = effect_descriptor(e);
    r = effect_any_reference(e);
    indices = reference_indices(r);
    if (!effect_region_p(e) || indices == NIL || descriptor_none_p(d)) {
      gen_remove(l, e);
    }
  }
}

/* Makes the body of a "write" loop.
   An assign statement is made using the array and the expression built using
   read variables. This allows to avoid dead code elimination.
*/
statement make_write_loopbody(entity v, expression exp, list vl) {
  expression e = reference_to_expression(make_reference(v, gen_full_copy_list(vl)));
  statement s = make_assign_statement(e, exp);
  pips_assert("write body is not properly generated", statement_consistent_p(s));
  return s;
}

/* Makes the body of a "read" loop.
   An assign statement is made using the variables generated and the array.
*/
statement make_read_loopbody(entity v,  entity readVar, list vl) {
  expression e = reference_to_expression(make_reference(v, gen_full_copy_list(vl)));
  expression e2 = entity_to_expression(readVar);
  statement s = make_assign_statement(e2, e);
  pips_assert("read body is not properly generated", statement_consistent_p(s));
  return s;
}

/* Builds a loop from a Psysteme.
   The same function is called whether it is a read or a write region, therefore the body changes.
   Please refer to the documentation of algorithm_row_echelon_generic and systeme_to_loop_nest for more
   details.
   Depending on whether it is a read or a write region, readVar is entity_undefined or exp is expression_undefined.
*/
static statement Psysteme_to_loop_nest(entity v,list vl, Pbase b, Psysteme p, bool isRead, entity readVar, expression exp, list l_var) {
  Psysteme condition, enumeration;
  statement body;
  if (isRead) 
    body = make_read_loopbody(v, readVar, l_var);
  else
    body = make_write_loopbody(v, exp, l_var);
  algorithm_row_echelon_generic(p, b, &condition, &enumeration, true);
  statement s = systeme_to_loop_nest(enumeration, vl, body, entity_intrinsic(DIVIDE_OPERATOR_NAME));
  pips_assert("s is not properly generated (systeme_to_loop_nest)", statement_consistent_p(s));
  return s;
}

/* Returns the entity corresponding to the global name */
static entity global_name_to_entity( const char* package, const char* name ) {
  return gen_find_tabulated(concatenate(package, MODULE_SEP_STRING, name, NULL), entity_domain);
}

/* Takes a region, its type (R/W), a read variable OR an expression (one of them is undefined).
   Returns a loop statement
   The scalar case has been made just in case, due to the filter, there should be no scalar variables left in the region list
*/
statement region_to_loop_nest (region r, bool isRead, entity readVar, expression exp) {
  reference ref = effect_any_reference(r);
  entity v = reference_variable(ref);
  type t = entity_type(v);
  statement s = statement_undefined; 
  if (type_variable_p(t)) {
    variable tv = type_variable(t);
    list dl = variable_dimensions(tv);
    if (ENDP(dl)) {
      s = make_nop_statement();
      append_comments_to_statement(s, "scalar case");
    }
    else {
      Psysteme p = region_system(r);
      Pbase base = BASE_NULLE;
      // Build the base
      FOREACH(expression, e, reference_indices(ref)) {
	entity phi = reference_variable(syntax_reference(expression_syntax(e)));
	base = base_add_variable(base, (Variable)phi);
      }
      s = Psysteme_to_loop_nest(v, base_to_list(base), base, p, isRead, readVar, exp, reference_indices(ref));
    }
    
  }
  else {
    pips_internal_error("unexpected type \n");
  }
  pips_assert("s is properly generated", statement_consistent_p(s));
  return s;
  
}

/* Makes an addition expression from two expressions */
expression make_addition(expression e1, expression e2) {
  entity add_ent = gen_find_entity("TOP-LEVEL:+");
  return make_call_expression(add_ent, CONS(EXPRESSION, e1, CONS(EXPRESSION, e2, NIL)));
}

/* This function is in charge of replacing the PHI entity of the region by generated indices.
   PHI values has no correspondance in the code. Therefore we have to create actual indices and
   replace them in the region in order for the rest to be build using the right entities.
*/
void replace_indices_region(region r, list* dadd, int indNum, entity module) {
  Psysteme ps = region_system(r);
  reference ref = effect_any_reference(r);
  list ref_indices = reference_indices(ref);
  list l_var = base_to_list(sc_base(ps));
  list l_var_new = NIL;
  list li = NIL;
  // Default name given to indices
  char* s = "REGIONS-PACKAGE:autogen";
  char s2[128];
  int indIntern = 0;
  list l_var_temp = gen_nreverse(gen_copy_seq(l_var));
  bool modified = false;
  // The objective here is to explore the indices and the variable list we got from the base in order to compare and
  // treat only the relevant cases
  FOREACH(entity, e, l_var_temp) {
    if (!ENDP(ref_indices)) {
      FOREACH(expression, exp, ref_indices) {
	entity phi = reference_variable(syntax_reference(expression_syntax(exp)));
	if (!strcmp(entity_name(phi), entity_name(e))) {
	  // If the names match, we generate a new name for the variable
	  sprintf(s2, "%s_%d_%d", s, indNum, indIntern);
	  indIntern++;
	  // We make a copy of the entity with a new name
	  entity ec = make_entity_copy_with_new_name(e, s2, false);
	  // However the new variable still has a rom type of storage, therefore we create a new ram object
	  entity dynamic_area = global_name_to_entity(module_local_name(module), DYNAMIC_AREA_LOCAL_NAME);
	  ram r =  make_ram(module, dynamic_area, CurrentOffsetOfArea(dynamic_area, e), NIL);
	  entity_storage(ec) = make_storage_ram(r);
	  s2[0] = '\0';
	  // We build the list we are going to use to rename the variables of our system
	  l_var_new = CONS(ENTITY, ec, l_var_new);
	  // We build the list which will replace the list of indices of the region's reference
	  li = CONS(EXPRESSION, entity_to_expression(ec), li);
	  // We build the list which will be used to build the declaration statement 
	  *dadd = CONS(ENTITY, ec, *dadd);
	  modified = true;
	}
      }
      if (!modified) {
	gen_remove_once(&l_var, e);
      }
    }
    modified = false;
  }
  pips_assert("different length \n", gen_length(l_var) == gen_length(l_var_new));
  // Renaming the variables of the system and replacing the indice list of the region's reference
  ps = sc_list_variables_rename(ps, l_var, l_var_new);
  reference_indices(ref) = gen_nreverse(gen_full_copy_list(li));
  pips_assert("region is not consistent", region_consistent_p(r));
}

/* Make a sequence from a statement list
   The equivalent of this function was already made somewhere else.
   However when the list only has one element it returns that element
   instead of making a sequence of one element containing that element.
   This function always makes a sequence
*/
statement make_sequence_from_statement_list(list l) {
  if (l == NIL) {
    return statement_undefined;
  }
  else {
    if (gen_length(l) == 1) {
      return make_block_with_stmt_if_not_already(STATEMENT(CAR(l)));
    }
    else {
      return make_block_statement(l);
    }
  }
}

/* This phase replaces the body of a function by automatically generated loops where the read
   and write statements are representative of the original function.
   If the regions are not computable then the function is not modified.
*/
bool regions_to_loops(char* module_name) {
  // List of the read/write regions
  list l_write = NIL;
  list l_read = NIL;
  
  // List of variables to add in the declaration statement
  list declarations_to_add = NIL;
  
  // Standard initialization
  entity module = local_name_to_top_level_entity(module_name);
  statement module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module);
  set_current_module_statement(module_stat);
  set_methods_for_convex_effects();
  
  // We fetch the summary region list
  list func_regions = effects_to_list((effects)db_get_memory_resource(DBR_SUMMARY_REGIONS, module_name, true));
  
  // And separate the R/W regions into two lists
  l_write = regions_write_regions(func_regions);
  l_read = regions_read_regions(func_regions);
  // We remove irrelevant regions from the lists
  filter_regions(&l_read);
  filter_regions(&l_write);
  
  // If no regions were fetched or if no region is left after filtering then we do nothing
  if (l_read == NIL && l_write == NIL) {
    // Standard reset
    reset_current_module_entity();
    reset_current_module_statement();  
    reset_out_summary_effects_list();
    reset_out_summary_regions_list();
    generic_effects_reset_all_methods();
    reset_ordering_to_statement();
    return true;
  }

  // List of statement we are going to use to build the new body of the function
  list sl = NIL;
  // Indices for variable and indices generation
  int varNum = 0;
  int indNum = 0;
  // List of the generated variables for read purposes
  list readVar = NIL;
  // a simple "3" in order to complete the "write" assign statements in case there is no read
  // Given that X1, X2 .. XN are the variables used to make the read statements
  // X1+X2+..+XN+3 is the expression used to make the write statements
  // In theory this avoids dead code elimination
  expression addVarRead = entity_to_expression(make_integer_constant_entity(3));
  // loop indice
  int i = 0;
  
  // We build as many variable as there are read regions 
  for (i = 0; i < gen_length(l_read); i++) {
    entity e = make_new_module_variable(get_current_module_entity(), varNum);
    // We add the new variable to the current expression used for write assignments
    addVarRead = make_addition(addVarRead, entity_to_expression(e));
    varNum++;
    // We add this variable to a specific list
    readVar = CONS(ENTITY, e, readVar);
  }
  // The variables for read purposes having been made, we can add them to the list of declarations to add
  declarations_to_add = gen_nconc(declarations_to_add, readVar);
  
  // Read regions processing
  // We replace the PHI variables of the region, we create a loop statement for this region and add
  // it to the list of statement of the new body
  FOREACH(effect, e, l_read) {
    replace_indices_region(e, &declarations_to_add, indNum, module);
    indNum++;
    statement s = region_to_loop_nest(e, true, gen_car(readVar), expression_undefined);
    POP(readVar);
    sl = CONS(STATEMENT, s, sl);
  }

  // Write regions processing
  // Same processing as above. 
  FOREACH(effect, e, l_write) {
    replace_indices_region(e, &declarations_to_add, indNum, module);
    indNum++;
    statement s = region_to_loop_nest(e, false, entity_undefined, addVarRead);
    sl = CONS(STATEMENT, s, sl);
  }

  // Inverting and duplicating the elements of the list in order to avoid corrupting data structures.
  sl = gen_full_copy_list(gen_nreverse(sl));
  // Make a new sequence statement in order to replace the old one
  statement ns = make_sequence_from_statement_list(sl);

  // Using the list of variables needed to be declared we call a function
  // appending declaration statements to our main sequence
  declarations_to_add = gen_full_copy_list(gen_nreverse(declarations_to_add));
  FOREACH(entity, e, declarations_to_add) {
    ns = add_declaration_statement(ns, e);
  }

  // Consistency check
  pips_assert("list of statement is not consistent", statement_consistent_p(ns));
  pips_assert("list of statement is not sequence", statement_sequence_p(ns));

  // Free of the old body
  free_statement(module_stat);

  // Reorder and putting resource for the new function body
  module_stat = ns;
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  
  // Standard reset
  reset_current_module_entity();
  reset_current_module_statement();  
  reset_out_summary_effects_list();
  reset_out_summary_regions_list();
  generic_effects_reset_all_methods();
  reset_ordering_to_statement();
  return true;
}
