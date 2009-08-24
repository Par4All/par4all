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

/* Flatten code

   Francois Irigoin, Fabien Coelho, Laurent Daverio.

 */
#define _GNU_SOURCE
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

// This function will move to ri-util/statement.c
void print_statements(list sl) 
{
  FOREACH(STATEMENT, s, sl) {
    print_statement(s);
  }
}



/* gen_multi_recurse callback on exiting a variable reference:

   if var present in renamings,
 */
static void rename_reference(reference r, hash_table renamings)
{
  entity var = reference_variable(r);
  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    pips_debug(1, "Reference %s renamed as %s\n",
	       entity_local_name(var), entity_local_name(nvar));
    reference_variable(r) = nvar;
  }
}

/* gen_multi_recurse callback on exiting a loop: rename loop index if
   appropriate.
 */
static void rename_loop_index(loop l, hash_table renamings)
{
  entity var = loop_index(l);
  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    pips_debug(1, "Loop index %s renamed as %s\n",
	       entity_local_name(var), entity_local_name(nvar));
    loop_index(l) = nvar;
  }
}

/* Should be moved into ri-util/variable.c */
expression variable_initial_expression(entity v)
{
  value val = entity_initial(v);
  expression exp = expression_undefined;

  if (value_expression_p(val)) {
    exp = copy_expression(value_expression(val));
  }
  else if(value_constant_p(val)) {
    constant c = value_constant(val);
    if (constant_int_p(c)) {
      exp = int_to_expression(constant_int(c));
    }
    else {
      pips_internal_error("Not Yet Implemented.\n");
    }
  }
  else if(value_code_p(val)) {
    if(pointer_type_p(ultimate_type(entity_type(v)))) {
      list il = sequence_statements(code_initializations(value_code(val)));

      if(!ENDP(il)) {
	statement is = STATEMENT(CAR(il));
	instruction ii = statement_instruction(is);

	pips_assert("A pointer initialization is made of one instruction expression",
		    gen_length(il)==1 && instruction_expression(ii));

	exp = copy_expression(instruction_expression(ii));
      }
    }
  }
  else if(value_unknown_p(val)) {
    exp = expression_undefined;
  }
  else {
    pips_internal_error("Unexpected value tag %d.\n", value_tag(val));
  }

  return exp;
}

/* gen_multi_recurse callback on exiting a statement:
 */
static void rename_statement_declarations(statement s, hash_table renamings)
{
  if (statement_block_p(s)) {

    list inits = NIL;
    list decls = statement_declarations(s); // Non-recursive
    instruction old = statement_instruction(s);
    list ndecls = NIL;

    FOREACH(ENTITY, var, decls) {
      entity nvar = (entity)hash_get(renamings, var);

      if(entity_undefined_p(nvar)) {
	pips_debug(1, "Local variable %s is preserved because its initial value "
		   "is not assignable\n", entity_local_name(var));
	ndecls = gen_nconc(ndecls, CONS(ENTITY, var, NIL));
      }
      else 
	/* If the new variable declaration does not contain the initial
	   value of the variable declaration, an initialization
	   statement must be inserted */
	if (!value_unknown_p(entity_initial(var))
	    && value_unknown_p(entity_initial(nvar))) {
	  expression ie = variable_initial_expression(var);
	  statement is = make_assign_statement(entity_to_expression(nvar), ie);

	  inits = gen_nconc(inits, CONS(statement, is, NIL));

	  pips_debug(1, "Initialize var %s with initial value of var %s: ",
		     entity_local_name(nvar), entity_local_name(var)
		     );
	  ifdebug(1){
	    print_expression(ie);
	    fprintf(stderr, "\n");
	  }
	}
    }

    ifdebug(1)
      print_statements(inits);

    // insert list of initialisation statements at the beginning of s
    inits = gen_nconc(inits, CONS(statement, instruction_to_statement(old), NIL));

    ifdebug(1)
      print_statements(inits);

    // insere une sequence
    statement_instruction(s) = make_instruction_sequence(make_sequence(inits));

    //gen_free_list(statement_declarations(s));

    statement_declarations(s) = ndecls;
    pips_debug(1, "Local declarations replaced.\n");

  }
}



/* Create a copy of an entity, with (almost) identical type, storage
   and initial value, but a slightly different name as entities are
   uniquely known by their names, and a different offset if the
   storage is ram.

   Entity e must be defined or the function core dumps.

   Depending on its storage, the new entity might have to be inserted
   in code_declarations and the memory allocation recomputed.

   Depending on the language, the new entity might have to be inserted
   in statement declarations. This is left up to the user of this function.

   @return the new entity.
*/
entity make_entity_copy_with_new_name(entity e, string global_new_name, bool move_initialization_p)
{
  entity ne = entity_undefined;
  char * variable_name = strdup(global_new_name);
  int number = 0;

  /* Find the first matching non-already existent variable name: */
  do {
    if (variable_name != NULL)
      /* Free the already allocated name in the previous iteration that
	 was conflicting: */
      free(variable_name);
    asprintf(&variable_name, "%s_%d", global_new_name, number++);
  }
  while(gen_find_tabulated(variable_name, entity_domain)
    != entity_undefined);

  //extended_integer_constant_expression_p(e)

  ne = make_entity(variable_name,
		   copy_type(entity_type(e)),
		   copy_storage(entity_storage(e)),
		   move_initialization_p? copy_value(entity_initial(e)) :
		   make_value_unknown()
		   );

  if(storage_ram_p(entity_storage(ne))) {
    /* We are in trouble. Up to now, we have created a static alias of
     * the variable e (it's a variable entity because of its
     * storage). Note that static aliases do not exist in C.
     */
    ram r = storage_ram(entity_storage(ne));
    entity m = ram_function(r);

    /* FI: It would be better to perorm the memory allocation right
       away, instead of waiting for a later core dump in chains or
       ricedg, but I'm in a hurry. */
    ram_offset(r) = UNKNOWN_RAM_OFFSET;

    AddEntityToDeclarations(ne, m);
  }
  return ne;
}

/* To generate the new variables, we need to know if there is an enclosing control cycle*/

typedef struct redeclaration_context {
  int cycle_depth;
  statement declaration_statement;
  string scope;
  string module_name;
  hash_table renamings;
} redeclaration_context_t;

static bool redeclaration_enter_statement(statement s, redeclaration_context_t * rdcp)
{
  instruction i = statement_instruction(s);

  /* Are we entering a (potential) cycle? Do we have a function to
     detect unstructured with no cycles? */
  if(instruction_loop_p(i)
     || instruction_whileloop_p(i)
     || instruction_forloop_p(i)
     || instruction_unstructured_p(i))
    rdcp->cycle_depth++;
  else if(instruction_sequence_p(i) && !ENDP(statement_declarations(s))) {
    FOREACH(ENTITY, v, statement_declarations(s)) {
      expression ie = variable_initial_expression(v);
      bool redeclare_p = FALSE;
      bool move_initialization_p = FALSE;

      /* Can we move or transform the initialization? */
      if(expression_undefined_p(ie)) {
	/* No initialization issue, let's move the declaration */
	redeclare_p = TRUE;
	move_initialization_p = TRUE;
      }
      else if(rdcp->cycle_depth>0) {
	/* We are in a control cycle. The initial value must be
	   reassigned where the declaration were. */
	if(expression_is_C_rhs_p(ie)) {
	  redeclare_p = TRUE;
	  move_initialization_p = FALSE;
	}
	else {
	  redeclare_p = FALSE;
	  move_initialization_p = FALSE;
	}
      }
      else {
	/* We are not in a control cycle. The initial value
	   expression, if constant, can be moved with the
	   new declaration. This avoids problem with non-assignale
	   expressions such as brace expressions used in
	   initializations at declaration. */
	if(extended_expression_constant_p(ie)) {
	  redeclare_p = TRUE;
	  move_initialization_p = TRUE;
	}
	else if(expression_is_C_rhs_p(ie)) {
	  redeclare_p = TRUE;
	  move_initialization_p = FALSE;
	}
	else {
	  redeclare_p = FALSE;
	  move_initialization_p = FALSE;
	}
      }

      if(redeclare_p) {
	string eun = entity_user_name(v);
	string mn = rdcp->module_name;

	string negn = strdup(concatenate(mn, MODULE_SEP_STRING, rdcp->scope, eun, NULL));
	entity nv = make_entity_copy_with_new_name(v, negn, move_initialization_p);

	statement_declarations(rdcp->declaration_statement) =
	  gen_nconc(statement_declarations(rdcp->declaration_statement),
		    CONS(ENTITY, nv, NIL));
	hash_put(rdcp->renamings, v, nv);
	pips_debug(1, "Variable %s renamed as %s\n", entity_name(v), entity_name(nv));
      }

    }
  }

  return TRUE;
}

static bool redeclaration_exit_statement(statement s,
				  redeclaration_context_t * rdcp)
{
  instruction i = statement_instruction(s);

  /* Are entering a (potential) cycle? */
  if(instruction_loop_p(i)
     || instruction_whileloop_p(i)
     || instruction_forloop_p(i)
     || instruction_unstructured_p(i))
    rdcp->cycle_depth--;

  return TRUE;
}

/* FI: added to wrap up the use of redeclaration context... */
static void compute_renamings(statement s, string sc, string mn, hash_table renamings)
{
  redeclaration_context_t rdc = { 0, s, sc, mn, renamings};

  gen_context_recurse(statement_instruction(s),
		      &rdc,
		      statement_domain,
		      redeclaration_enter_statement,
		      redeclaration_exit_statement);

}

/*
  This functions locates all variable declarations in embedded blocks,
  and moves them to the top-level block, renaming them in case of conflicts.

  First, we are going to loop through each declaration in the
  statement and its sub-blocks, and build two collections :
  
  - variables: a set to keep track of the variable declarations
    already encountered

  - renamings: an (entity-> new entity) hash of pointers to keep track
    of renamed variables

  If a variable was already encountered (i.e. already in the set), we
  probably have a naming conflict. In that case, we create a new
  entity sharing the same properties as the conflicting one, but with
  a derived name (original name + numerical suffix), and we update the
  set and the hashtable with the new entity. In the absence of a
  conflict, though, we just add the variable name to the set.

  When the loop is done, we can then use these collections to update
  the statement via a gen_multi_recurse. Specifically, we need to:

  - rename variable references
  - rename loop indexes
  - replace declaration statements       
*/

void statement_flatten_declarations(statement s)
{
  /* For the time being, we handle only blocks with declarations */
  if (statement_block_p(s) && !ENDP(statement_declarations(s))) {

    list declarations = instruction_to_declarations(statement_instruction(s)); // Recursive

    hash_table renamings = hash_table_make(hash_pointer, 10);

    entity se   = ENTITY(CAR(statement_declarations(s)));
    string sen  = entity_name(se);
    string seln = entity_local_name(se);
    string cs   = local_name_to_scope(seln); /* current scope for s */
    string mn   = module_name(sen);

    compute_renamings(s, cs, mn, renamings);

    /*
    FOREACH(ENTITY, e, declarations) {

      string eun = entity_user_name(e);

      string negn = strdup(concatenate(mn, MODULE_SEP_STRING, cs, eun, NULL));
      entity ne = make_entity_copy_with_new_name(e, negn);

      statement_declarations(s) = gen_nconc(statement_declarations(s), CONS(ENTITY, ne, NIL));
      hash_put(renamings, e, ne);
      pips_debug(1, "Variable %s renamed as %s\n", entity_name(e), entity_name(ne));

    }
    */

    ifdebug(1)
      hash_table_fprintf(stderr, entity_local_name, entity_local_name, renamings);

    //char *(*key_to_string)(void*),
    //char *(*value_to_string)(void*),

    pips_debug(1, "gen_multi_recurse\n");

    gen_context_multi_recurse( statement_instruction(s), renamings, 
			       reference_domain, gen_true, rename_reference,
			       loop_domain, gen_true, rename_loop_index,
			       statement_domain, gen_true, rename_statement_declarations,
			       NULL );
    

    gen_free_list(declarations), declarations = NIL;
    hash_table_free(renamings), renamings = NULL;
    // call sequence flattening as all declarations are now global
    clean_up_sequences(s);
  }

  else
    pips_internal_error("Input assumptions not met.\n");
}


static bool unroll_loops_in_statement(statement s) {

  if (statement_loop_p(s)) {
    loop l = statement_loop(s);
    
    if (loop_fully_unrollable_p(l)) 
      full_loop_unroll(s);
  }
  return TRUE;
}


/* This function is/will be composed of several steps:
   
   - flatten declarations inside statement

 */
bool flatten_code(string module_name)
{
  entity module;
  statement module_stat;

  //bool res;

  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();
 
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, TRUE) );
  module_stat = get_current_module_statement();

  debug_on("FLATTEN_CODE_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  
  /* Step 1: flatten declarations */
  statement_flatten_declarations(module_stat);

  /* Step 2: unroll loops */
  gen_recurse( module_stat,
	       statement_domain, gen_true, unroll_loops_in_statement
	       );
  clean_up_sequences(module_stat);

  // This might not really be necessaty
  module_reorder(module_stat);

  pips_debug(1, "end\n");
  debug_off();

  

  /* Save modified code to database */
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

  reset_current_module_entity();
  reset_current_module_statement();

  /* Return value */
  bool good_result_p = TRUE;
  
  return (good_result_p);
}
