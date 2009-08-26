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


/* gen_multi_recurse callback on exiting a variable reference:
   if var needs renaming, rename this reference.
 */
static void rename_reference(reference r, hash_table renamings)
{
  entity var = reference_variable(r);
  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    if(nvar!=var) {
      pips_debug(1, "Reference %s renamed as %s\n",
		 entity_local_name(var), entity_local_name(nvar));
      reference_variable(r) = nvar;
    }
  }
}

/* gen_multi_recurse callback on exiting a loop:
   if loop index needs renaming, rename this occurrence.
 */
static void rename_loop_index(loop l, hash_table renamings)
{
  entity var = loop_index(l);
  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    if(nvar!=var) {
      pips_debug(1, "Loop index %s renamed as %s\n",
		 entity_local_name(var), entity_local_name(nvar));
      loop_index(l) = nvar;
    }
  }
}

/* gen_multi_recurse callback on exiting a statement:recompute the
   declaration list for statement s and transform initializations into
   assignments when required according to the renaming map
   "renamings". Renaming may be neutral to handle external
   variables. The initial values are used to specify if an assignment
   must be created or not. */
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
      else if(var!=nvar) {
	/* If the new variable declaration does not contain the
	   initial value of the variable declaration, an
	   initialization statement must be inserted */
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
      else {
	/* Do nothing and the local declaration will be lost */
	pips_debug(1, "Declaration for external variable \"%s\" moved.\n",
		   entity_name(var));
      }
    }

    /* Insert the list of initialisation statements as a sequence at
       the beginning of s.
    */
    inits = gen_nconc(inits, CONS(statement, instruction_to_statement(old), NIL));
    ifdebug(1)
      print_statements(inits);
    statement_instruction(s) = make_instruction_sequence(make_sequence(inits));

    //gen_free_list(statement_declarations(s));

    statement_declarations(s) = ndecls;
    pips_debug(1, "Local declarations replaced.\n");
  }
}

/* To generate the new variables, we need to know:
 *
 *  - if there is an enclosing control cycle
 *
 *  - what is the (current) statement to be used for declaration
 *
 *  - the current scope corresponding to that statement
 *
 *  - the current module name (get_current_module_name() could be used
 *  instead)
 *
 *  - and the renaming map
 *
 * This data structure is private to flatten_code.c
 */
typedef struct redeclaration_context {
  int cycle_depth;
  statement declaration_statement;
  string scope;
  string module_name;
  hash_table renamings;
} redeclaration_context_t;

/* This function makes the key decision about the renaming: should
   the variable be renamed? Is the renaming and declaration move
   compatible with its initialization expression and its control
   context? */
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
      string mn = rdcp->module_name;
      string vn = entity_name(v);
      string vmn = module_name(vn);

      if(strcmp(mn, vmn)!=0) {
	/* This is not a local variable. Its declaration can be
	   moved if not already there. */
	statement ds = rdcp->declaration_statement;
	list dv = statement_declarations(ds);

	if(!entity_is_argument_p(v, dv)) {
	  statement_declarations(ds) = gen_nconc(dv, CONS(ENTITY, v, NIL));
	}
	hash_put(rdcp->renamings, v, v);
      }
      else { /* This is a block local stack allocated or static
		variable */
	/* FI: the case of static variables is not taken into account
	   properly. */
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
	     reassigned where the declaration was, if the variable is
	     not static. */
	  if(variable_static_p(v)) {
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
	  /* Build the new variable */
	  string eun = entity_user_name(v);

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
  }

  return TRUE;
}

/* Keep track of cycle exit in the hierarchical control flow graph */
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
  string mnc = strdup(mn);
  redeclaration_context_t rdc = { 0, s, sc, mnc, renamings};

  gen_context_recurse(statement_instruction(s),
		      &rdc,
		      statement_domain,
		      redeclaration_enter_statement,
		      redeclaration_exit_statement);
  free(mnc);
}

/*
  This functions locates all variable declarations in embedded blocks,
  and moves them to the top-level block when possible, renaming them
  in case of conflicts.

  First, we are going to loop through each declaration in the
  statement instruction (no in the statement itself) and its
  sub-blocks, and build a renaming map: an (entity-> new entity) hash
  of pointers to keep track of renamed variables

  Not all variable declarations can be moved and/or renamed. Not all
  initializations can be transformed into assignments. And some
  variables declared locally are not variables local to the block.

  If a variable with the name we would like for the renamed variable
  is already in the symbol table, we have a naming conflict. In that case, we
  create a new entity sharing the same properties as the conflicting
  one, but with a derived name (original name + numerical suffix), and
  we update the hashtable with the new entity

  When the renaming map is computed, we can then use it to update
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
    bool renaming_p = FALSE;

    /* Can we find out what the local scope os statement s is? */
    FOREACH(ENTITY, se, statement_declarations(s)) {
      string sen  = entity_name(se);
      string seln = entity_local_name(se);
      string cs   = local_name_to_scope(seln); /* current scope for s */
      string mn   = module_name(sen);
      string cmn = entity_user_name(get_current_module_entity());

      if(strcmp(mn, cmn)==0) {
	compute_renamings(s, cs, mn, renamings);
	renaming_p = TRUE;
	break;
      }
    }

    if(renaming_p) {
      ifdebug(1)
	hash_table_fprintf(stderr,
			   // The warning will disappear when Fabien
			   // updates Newgen
			   //(char * (*)(void *)) entity_local_name,
			   //(char * (*)(void *)) entity_local_name,
			   (gen_string_func_t) entity_local_name,
			   (gen_string_func_t) entity_local_name,
			   renamings);

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
    }
    else {
      pips_user_warning("Code flattening fails because the statement does"
			" not contain any local declaration\n");
    }
    // call sequence flattening as some declarations may have been
    // moved up
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


/* This function is be composed of several steps:

   - flatten declarations inside statement: declarations are moved as
     high as possible in the control structure

   - clean_up_sequences: remove useless braces when they are nested

   - unroll looops with statically known iteration number

   The first two steps have been merged.

   It is assumed that the function main statement will contain at
   least one local variable. This is used to preserve the scoping
   mechanism used by the parser. Thus, "void foo(void){{{}}}" cannot
   be flatten. Note that clean_up_sequences could be used first to
   avoid such cases. Funcion "void foo(void){{{extern int i;}}}"
   cannot be flatten either, but clean_up_sequences might help.

 */
bool flatten_code(string module_name)
{
  entity module;
  statement module_stat;
  bool good_result_p = TRUE;

  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();

  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, TRUE) );
  module_stat = get_current_module_statement();

  debug_on("FLATTEN_CODE_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  /* Step 1: flatten declarations and clean-up sequences */
  statement_flatten_declarations(module_stat);

  /* Step 2: unroll loops */
  gen_recurse( module_stat,
	       statement_domain, gen_true, unroll_loops_in_statement
	       );
  clean_up_sequences(module_stat); // again

  // This might not really be necessary, probably thanks to clean_up_sequences
  module_reorder(module_stat);

  pips_debug(1, "end\n");
  debug_off();

  /* Save modified code to database */
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

  reset_current_module_entity();
  reset_current_module_statement();

  return (good_result_p);
}
