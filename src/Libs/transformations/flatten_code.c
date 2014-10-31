/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

/* Flatten code

   Francois Irigoin, Fabien Coelho, Laurent Daverio.

 */
#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformations.h"


/* gen_multi_recurse callback on exiting a variable reference:
   if var needs renaming, rename this reference.
 */
static void rename_reference(reference r, hash_table renamings)
{
  entity var = reference_variable(r);
  bool replaced = false; // Keep track of any replacement
  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    if(nvar!=var) {
      pips_debug(1, "Reference %s renamed as %s\n",
		 entity_local_name(var), entity_local_name(nvar));
      reference_variable(r) = nvar;
      replaced = true;
    }
  }

  if(replaced) {
    /* we need to unormalize the uppermost parent of this expression
     * otherwise its normalized field gets incorrect */
    expression next=(expression)r,parent = NULL;
    while((next=(expression) gen_get_ancestor(expression_domain,next))) {
      parent=next;
    }
    if(parent) {
      unnormalize_expression(parent); /* otherwise field normalized get wrong */
    }
  }
}

/* gen_multi_recurse callback on exiting a loop:
 * if loop index needs renaming, rename this occurrence.
 *
 * Take advantage of this opportunity to serialize the loop in order
 * to avoid any inconsistency. Local variables moved out of the loop
 * may require a privatization after flattening of the loop is to be
 * kept parallel.
 */
static void rename_loop_index(loop l, hash_table renamings)
{
  entity var = loop_index(l);

  execution ex = loop_execution(l);
  if(execution_parallel_p(ex))
    execution_tag(ex) = is_execution_sequential;

  if (hash_defined_p(renamings, var)) {
    entity nvar = (entity)hash_get(renamings, var);
    if(nvar!=var) {
      pips_debug(1, "Loop index %s renamed as %s\n",
		 entity_local_name(var), entity_local_name(nvar));
      loop_index(l) = nvar;
    }
  }
}

/* If the type of variable var is a typedefed type, it may have been
   renamed and the symbol table must be updated. */
void rename_variable_type(entity var, hash_table renamings)
{
  type t = entity_type(var);
  if(typedef_type_p(t)) {
    variable v = type_variable(t);
    basic b = variable_basic(v);
    entity tt = basic_typedef(b);
    entity ntt = (entity)hash_get(renamings, tt);
    if(!entity_undefined_p(ntt))
      basic_typedef(b) = ntt;
  }
  else if(struct_type_p(t) || union_type_p(t)){
    variable v = type_variable(t);
    basic b = variable_basic(v);
    entity tt = basic_derived(b);
    entity nst = (entity)hash_get(renamings, tt);
    if(!entity_undefined_p(nst))
      basic_typedef(b) = nst;
  }
}

/* gen_multi_recurse callback on exiting a statement: recompute the
   declaration list for statement s and transform initializations into
   assignments when required according to the renaming map
   "renamings". Renaming may be neutral to handle external
   variables. The initial values are used to specify if an assignment
   must be created or not. */
static void rename_statement_declarations(statement s, hash_table renamings)
{
  if (declaration_statement_p(s)) {
    list inits = NIL;
    list decls = statement_declarations(s); // Non-recursive
    instruction old = statement_instruction(s);
    list ndecls = NIL;
    list tmp = NIL; /* holds the entity to remove from declarations */

    pips_debug(1, "Begin for statement %p\n", s);

    FOREACH(ENTITY, var, decls) {
      entity nvar = (entity)hash_get(renamings, var);

      if(entity_undefined_p(nvar)) {
	/* Well, we could synthesize a new function to perform the
	   initialization. */
	pips_debug(1, "Local variable %s is preserved because its initial value "
		   "is not assignable\n", entity_local_name(var));
	ndecls = gen_nconc(ndecls, CONS(ENTITY, var, NIL));
	replace_entities(entity_type(var),renamings);
      }
      else if(var!=nvar) {
	/* If the new variable declaration does not contain the
	   initial value of the variable declaration, an
	   initialization statement must be inserted */
	if (!value_unknown_p(entity_initial(var))
	    && value_unknown_p(entity_initial(nvar))) {
	  expression ie = variable_initial_expression(var);
	  statement is = make_assign_statement(entity_to_expression(nvar), ie);
	  gen_context_recurse(ie,renamings,
			      reference_domain, gen_true, rename_reference);
	  replace_entities(entity_type(nvar),renamings);

	  inits = gen_nconc(inits, CONS(statement, is, NIL));

	  pips_debug(1, "Initialize var %s with initial value of var %s: ",
		     entity_local_name(nvar), entity_local_name(var)
		     );
	  ifdebug(1){
	    print_expression(ie);
	    fprintf(stderr, "\n");
	  }
	}
	tmp=CONS(ENTITY,var,tmp);
      }
      else {
	/* FI: The comment below used to be true before we used
	   declaration statements... */
	/* Do nothing and the local declaration will be lost */
	pips_debug(1, "Declaration for external variable \"%s\" moved.\n",
		   entity_name(var));
      }
      /* Should we worry that the type itself has been renamed because
	 a typedef is used? */
      rename_variable_type(var, renamings);
    }

    /* calling RemoveLocalEntityFromDeclarations will tidy the
       declarations and the declaration_statements */
    FOREACH(ENTITY,e,tmp)
      RemoveLocalEntityFromDeclarations(e,get_current_module_entity(),s);
    gen_free_list(tmp);

    if(!ENDP(inits)) {
      /* Insert the list of initialisation statements as a sequence at
	 the beginning of s.
      */
      inits = gen_nconc(inits,
			CONS(statement, instruction_to_statement(old), NIL));
      ifdebug(1)
	print_statements(inits);
#if 0
      if(get_bool_property("C89_CODE_GENERATION")) {
	/* The initializations must be inserted at the right place,
	   which may prove impossible if some of the initializations
	   cannot be moved but are used. Example:

	   int a[] = {1, 2, 3};
	   int i = a[1];
	*/
	pips_internal_error("C89 flattened code not generated yet");
      }
      else
#endif
      { /* C99*/
	statement_instruction(s) =
	  make_instruction_sequence(make_sequence(inits));
	/* FI: why kill he initial statement number? */
	statement_number(s)=STATEMENT_NUMBER_UNDEFINED;
	if(!statement_with_empty_comment_p(s)) {
	  string c = statement_comments(s);
	  statement fs = STATEMENT(CAR(inits));
	  statement_comments(fs) = c;
         /* FI: should be a call to defined_empty_comments() or
            something like it. Currently, empty_comments is a macro
            and its value is string_undefined:-( */
	  statement_comments(s) = strdup("");
	}
      }
    }

    //gen_free_list(statement_declarations(s));

    statement_declarations(s) = ndecls;

    pips_debug(1, "End. Local declarations %s.\n",
	       ENDP(ndecls)? "removed" : "updated");
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
  const char* scope;
  const char* module_name;
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
      const char* mn = rdcp->module_name;
      string vn = entity_name(v);
      const char* vmn = module_name(vn);

      if (hash_defined_p(rdcp->renamings, v)) {
	pips_debug(5, "Skipping the already processed variable \"%s\" \n",
		   entity_user_name(v));
	continue;
      }

      if(strcmp(mn, vmn)!=0) {
	/* This is not a local variable. Its declaration can be
	   moved if not already there. */
	statement ds = rdcp->declaration_statement;
	list dv = statement_declarations(ds);

	if(!entity_is_argument_p(v, dv)) {
	  pips_debug(5, "Entity is not an argument\n");
	  AddLocalEntityToDeclarations(v,get_current_module_entity(),ds);
	}
	hash_put_or_update(rdcp->renamings, v, v);
      }
      else { /* This is a block local stack allocated or static
		variable */
	/* FI: the case of static variables is not taken into account
	   properly. */
	expression ie = variable_initial_expression(v);
	bool redeclare_p = false;
	bool move_initialization_p = false;

	/* Can we move or transform the initialization? */
	if(expression_undefined_p(ie) || entity_static_variable_p(v)) {

	  /* No initialization issue, let's move the declaration */
	  redeclare_p = true;
	  move_initialization_p = true;
	}
	else if(rdcp->cycle_depth>0) {
	  /* We are in a control cycle. The initial value must be
	     reassigned where the declaration was, if the variable is
	     not static. */
	  if(variable_static_p(v)) {
	    redeclare_p = true;
	    move_initialization_p = true;
	  }
	  else if(expression_is_C_rhs_p(ie)) { // This function is not yet precise enough
	    redeclare_p = true;
	    move_initialization_p = false;
	  }
	  else {
	    /* It could be redeclared if a small function was
	       synthesized to perform the assignment
	       dynamically. Basically, a loop nest over the array
	       dimensions. */
	    redeclare_p = false;
	    move_initialization_p = false;
	  }
	}
	else {
	  /* We are not in a control cycle. The initial value
	     expression, if constant, can be moved with the
	     new declaration. This avoids problem with non-assignable
	     expressions such as brace expressions used in
	     initializations at declaration. */
	  if(extended_expression_constant_p(ie)) {
	    redeclare_p = true;
	    move_initialization_p = true;
	  }
	  else if(expression_is_C_rhs_p(ie)) {
	    redeclare_p = true;
	    move_initialization_p = false;
	  }
	  else {
	    redeclare_p = false;
	    move_initialization_p = false;
	  }
	}

	if(redeclare_p) {

	  /* Build the new variable */
	  const char* eun  = entity_user_name(v);
	  string negn = typedef_entity_p(v)?
	    strdup(concatenate(mn, MODULE_SEP_STRING, rdcp->scope, 
			       TYPEDEF_PREFIX, eun, NULL))
	    :
	    strdup(concatenate(mn, MODULE_SEP_STRING, rdcp->scope, eun, NULL));
	  entity nv   = entity_undefined;
	  //list unused_nvs = NIL;

	  /* When renaming the variable, we must make sure that we are
	     not creating a user name conflict at source-code
	     level. For now we will keep regenerating nv and checking
	     it against the list of all entities used in the
	     statement, until no conflict remains.
	  */

	  statement ds = rdcp->declaration_statement;
	  list dselist = statement_to_referenced_entities(ds);
	  bool is_same_name    = false;

	  ifdebug(8) {
	    pips_debug(8, "Entities found in declaration statement: ");
	    print_entities(dselist);
	    fprintf(stderr, "\n");
	  }

	  /* We iterate over suffixes (_0, _1, _2, ...) and test if we
	     generate a conflict */
	  do {
	    nv = make_entity_copy_with_new_name(v, negn, move_initialization_p);
	    FOREACH(ENTITY, dv, dselist) {

	      if (dv == v)
	      {
		pips_debug(8, "Skipping the variable \"%s\" we are working on\n",
			   entity_user_name(v));
		continue;
	      }

	      is_same_name =
		strcmp(entity_user_name(dv), entity_user_name(nv)) == 0;
	      if (is_same_name) {
		pips_debug(1, "Proposed variable \"%s\" "
			   "conflicts with references in declaration statement\n",
			   entity_name(nv));
		break;
	      }
	    }
	    if (is_same_name) {
	      // WARNING: We must remember to free the newly declared nv when it's not used!
	      //unused_nvs = CONS(ENTITY, nv, unused_nvs);
	    }
	  } while (is_same_name);

	  /* FI: what happens to external entities whose declarations
	     is moved, but the name unchanged? */
	  AddLocalEntityToDeclarations(nv,
				       get_current_module_entity(),
				       rdcp->declaration_statement);
	  hash_put_or_update(rdcp->renamings, v, nv);
	  pips_debug(1, "Variable %s renamed as %s\n", entity_name(v), entity_name(nv));
	}
      }
    }
  }

  return true;
}

/* Keep track of cycle exit in the hierarchical control flow graph */
static void redeclaration_exit_statement(statement s,
				  redeclaration_context_t * rdcp)
{
  instruction i = statement_instruction(s);

  /* Are entering a (potential) cycle? */
  if(instruction_loop_p(i)
     || instruction_whileloop_p(i)
     || instruction_forloop_p(i)
     || instruction_unstructured_p(i))
    rdcp->cycle_depth--;
}

/* FI: added to wrap up the use of redeclaration context... */
static void compute_renamings(statement s, const char* sc, const char* mn, hash_table renamings)
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

  @return false if the initial condition are not met, i.e. no parent block

  FIXME : the return false is not implemented !
*/
bool statement_flatten_declarations(entity module, statement s)
{
  /* For the time being, we handle only blocks with declarations */
    if (statement_block_p(s)) {
        list declarations = instruction_to_declarations(statement_instruction(s)); // Recursive
        hash_table renamings = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);
        bool renaming_p = false;

        /* Can we find out what the local scope of statement s is? */
        FOREACH(ENTITY, se, entity_declarations(module)) {
            string sen  = entity_name(se);
            const char* seln = entity_local_name(se);
            string cs   = local_name_to_scope(seln); /* current scope for s */
            const char* mn   = module_name(sen);
            const char* cmn = entity_user_name(module);

            if(same_string_p(mn, cmn)) {
                compute_renamings(s, cs, mn, renamings);
                renaming_p = true;
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

            gen_context_multi_recurse( statement_instruction(s), renamings,
                    reference_domain, gen_true, rename_reference,
                    loop_domain, gen_true, rename_loop_index,
                    statement_domain, gen_true, rename_statement_declarations,
                    NULL );

            // This look like a bad hack, partially redundant with previous replacement
            // but... only partially ! For instance extensions was not handled previously.
            // Probably that there's need for factoring, but it'll be another time !
            replace_entities(s,renamings);

            gen_free_list(declarations), declarations = NIL;
            hash_table_free(renamings), renamings = NULL;
        }
        else {
            pips_debug(2,"Code flattening fails because the statement does"
                    " not contain any local declaration\n");
        }
    }
    return true;
}


static bool unroll_loops_in_statement(statement s) {

  if (statement_loop_p(s)) {
    loop l = statement_loop(s);

    if (loop_fully_unrollable_p(l))
      full_loop_unroll(s);
  }
  return true;
}

static void statement_purge_declarations_walker(sequence seq)
{
    statement block = (statement)gen_get_ancestor(statement_domain,seq);
    list decls = gen_copy_seq(statement_declarations(block));

    FOREACH(ENTITY,e,decls)
    {
        bool decl_stat_found = false;
        FOREACH(STATEMENT,s,sequence_statements(seq))
        {
            if(( decl_stat_found = ( declaration_statement_p(s) && !gen_chunk_undefined_p(gen_find_eq(e,statement_declarations(s))) ) ) )
                break;
        }
        if(!decl_stat_found)
            gen_remove_once(&statement_declarations(block),e);
    }
    gen_free_list(decls);
}

static void statement_purge_declarations(statement s)
{
    gen_recurse(s,sequence_domain,gen_true,statement_purge_declarations_walker);
}


/* Pipsmake 'flatten_code' phase.

   This function is be composed of several steps:

   1 flatten declarations inside statement: declarations are moved as
     high as possible in the control structure; this may serialize
     parallel loops, but this pass was designed for sequential code.

   2 clean_up_sequences: remove useless braces when they are nested.

   3 unroll looops with statically known iteration number.

   4 clean_up_sequences: remove useless braces when they are nested.

   It is assumed that the function main statement will contain at
   least one local variable. This is used to preserve the scoping
   mechanism used by the parser. Thus, "void foo(void){{{}}}" cannot
   be flatten. Note that clean_up_sequences could be used first to
   avoid such cases. Function "void foo(void){{{extern int i;}}}"
   cannot be flatten either, but clean_up_sequences might help.

 */
bool flatten_code(const char* module_name)
{
  entity module;
  statement module_stat;
  bool good_result_p = true;

  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();

  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();

  debug_on("FLATTEN_CODE_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  // Step 1 and 2: flatten declarations and clean up sequences
  if ((good_result_p=statement_flatten_declarations(module,module_stat)))
  {
    statement_purge_declarations(module_stat);
    // call sequence flattening as some declarations may have been moved up
    clean_up_sequences(module_stat);

    // Step 3 and 4: unroll loops and clean up sequences
    if(get_bool_property("FLATTEN_CODE_UNROLL"))
    {
      gen_recurse(module_stat,
                  statement_domain, gen_true, unroll_loops_in_statement);
      clean_up_sequences(module_stat); // again
    }

    // This might not be necessary, thanks to clean_up_sequences
    module_reorder(module_stat);
    // Save modified code to database
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);
  }

  pips_debug(1, "end\n");
  debug_off();

  reset_current_module_entity();
  reset_current_module_statement();

  return (good_result_p);
}


/* Recurse through the statements of s and split local declarations.
   For the time being, we handle only blocks with declarations.

   NOTE: Statement s is modified in-place.

   This function can be called from another module to apply
   transformation directly.
*/
void statement_split_initializations(statement s)
{
    gen_recurse(s, statement_domain, gen_true, split_initializations_in_statement);
    /* Is it still useful? */
    clean_up_sequences(s);
}

/* Pipsmake 'split_initializations' phase
 */
bool split_initializations(const char* module_name)
{
  statement module_stat;
  bool good_result_p = true;

  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();

  debug_on("SPLIT_INITIALIZATIONS_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  // Do split !
  statement_split_initializations(module_stat);

  pips_debug(1, "end\n");
  debug_off();

  /* Save modified code to database */
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

  reset_current_module_entity();
  reset_current_module_statement();

  return (good_result_p);
}


void split_update_call(call c)
{
    entity op = call_function(c);
    list args = call_arguments(c);
    entity new_op = update_operator_to_regular_operator(op);
    if(!entity_undefined_p(new_op))
    {
        if(ENTITY_PLUS_C_P(new_op)||ENTITY_MINUS_C_P(new_op))
        {
            bool has_pointer =false;
            FOREACH(EXPRESSION,exp,call_arguments(c))
            {
                basic b = basic_of_expression(exp);
                if(basic_pointer_p(b)) { has_pointer=true;}
                free_basic(b);
            }
            if(!has_pointer) {
                if(ENTITY_PLUS_C_P(new_op))new_op=entity_intrinsic(PLUS_OPERATOR_NAME);
                else new_op=entity_intrinsic(MINUS_OPERATOR_NAME);
            }
        }
        ifdebug(1){
            expression tmp = call_to_expression(c);
            pips_debug(1,"changed expression \n");
            print_expression(tmp);
            syntax_call(expression_syntax(tmp))=call_undefined;
            free_expression(tmp);
        }

        call_function(c)=entity_intrinsic(ASSIGN_OPERATOR_NAME);
        expression lhs = binary_call_lhs(c);
        expression rhs = binary_call_rhs(c);
        CAR(CDR(args)).p=(gen_chunkp)MakeBinaryCall(
                new_op,
                copy_expression(lhs),
                rhs);

        ifdebug(1){
            expression tmp = call_to_expression(c);
            pips_debug(1,"into expression \n");
            print_expression(tmp);
            syntax_call(expression_syntax(tmp))=call_undefined;
            free_expression(tmp);
        }
    }
}

static void
split_update_operator_statement_walker(statement s)
{
  /* FI: this should be guarded by a declaration_statement_p(s),
     shouldn't it? */
  if(declaration_statement_p(s)) {
    FOREACH(ENTITY,e,statement_declarations(s))
    {
        value v = entity_initial(e);
              if( !value_undefined_p(v) && value_expression_p( v ) )
                 gen_recurse(v,call_domain,gen_true, split_update_call);
    }
  }
}

bool split_update_operator(const char* module_name)
{
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement( (statement)	db_get_memory_resource(DBR_CODE, module_name, true) );
  debug_on("SPLIT_UPDATE_OPERATOR_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  gen_multi_recurse(get_current_module_statement(),
          call_domain,gen_true,split_update_call,
          statement_domain,gen_true,split_update_operator_statement_walker,
          NULL);

  pips_debug(1, "end\n");
  debug_off();

  /* Save modified code to database */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}
