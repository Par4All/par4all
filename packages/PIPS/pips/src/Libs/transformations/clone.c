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
/*
 * Cloning of a subroutine.
 * debug: CLONE_DEBUG_LEVEL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "pipsmake.h"
#include "properties.h"
#include "preprocessor.h"
#include "semantics.h"
#include "callgraph.h"


#define DEBUG_ON   debug_on("CLONE_DEBUG_LEVEL")
#define ALL_DECLS  "PRETTYPRINT_ALL_DECLARATIONS"
#define STAT_ORDER "PRETTYPRINT_STATEMENT_NUMBER"
#define undefined_number_p(n) ((n)==STATEMENT_NUMBER_UNDEFINED)



/************************************************** BUILD THE CLONE VERSIONS */

/* build a new clone version. if argn is not null, generate a check.
 */
static statement
build_statement_for_clone(
    entity cloned, 
    int argn, 
    int val)
{
    statement stat, check_arg_value;

    if (argn>0)
    {
	entity arg = find_ith_parameter(cloned, argn);

	/* IF (PARAM#ARGN.NE.VAL) STOP
	 */
	check_arg_value = test_to_statement
	   (make_test(MakeBinaryCall(entity_intrinsic(NON_EQUAL_OPERATOR_NAME),
	     entity_to_expression(arg), int_to_expression(val)),
		       call_to_statement(make_call(entity_intrinsic
						   (STOP_FUNCTION_NAME), NIL)),
		       make_continue_statement(entity_undefined)));

	statement_comments(check_arg_value) =
	  strdup(concatenate(fortran_module_p(get_current_module_entity())?
			     "!! PIPS: " : "// PIPS: ",
			     entity_user_name(arg),
			     " is assumed a constant reaching value\n", NULL));
    }
    else
	check_arg_value = make_continue_statement(entity_undefined);

    stat = make_block_statement(
            make_statement_list(check_arg_value, copy_statement(get_current_module_statement()))
            );

    return stat;
}

/* create an clone, and returns the corresponding entity,
 * which looks like a not yet parsed routine.
 * it puts the initial_file for the routine, and updates its user_file.
 */
static entity build_a_clone_for(entity cloned,
				int argn,
				int val)
{
  const char* name = entity_local_name(cloned);
  char* comments, *new_name;
  entity new_fun;
  statement stat;
  bool saved_b1, saved_b2;
  text t;
  language l = module_language(cloned);
  
  const char* suffix = get_string_property("RSTREAM_CLONE_SUFFIX");
  int size = 0;

  pips_debug(2, "building a version of %s with arg %d val=%d\n",
	     name, argn, val);

  /* builds some kind of module / statement for the clone.
   */
  if (empty_string_p(suffix)) {
    const char *clone_name = get_string_property("CLONE_NAME");
    new_name = empty_string_p(clone_name) ?
      build_new_top_level_module_name(name,false) :
      strdup(clone_name);
  }
  else {
    size = strlen(name) + strlen(suffix) + 1;
    new_name = (char*)malloc(sizeof(char)*size);
    new_name[0] = '\0';
    new_name = strcat(new_name, name);
    new_name = strcat(new_name, "_");
    new_name = strcat(new_name, suffix);
  }

  new_fun = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,new_name);
  entity_type(new_fun) = copy_type(entity_type(cloned));
  entity_storage(new_fun) = copy_storage(entity_storage(cloned));
  entity_initial(new_fun) = entity_initial(cloned); /* no copy, reseted later*/

  saved_b1 = get_bool_property(ALL_DECLS);
  saved_b2 = get_bool_property(STAT_ORDER);
  set_bool_property(ALL_DECLS, true);
  set_bool_property(STAT_ORDER, false);

  stat = build_statement_for_clone(cloned, argn, val);
  t = text_named_module(new_fun, cloned, stat);


  set_bool_property(ALL_DECLS, saved_b1);
  set_bool_property(STAT_ORDER, saved_b2);

  /* add some comments before the code.
   */
  char *comment_prefix = fortran_module_p(cloned) ? "!!" : "//";
  comments = strdup(concatenate(
				comment_prefix,"\n",
				comment_prefix," PIPS: please caution!\n",
				comment_prefix,"\n",
				comment_prefix," this routine has been generated as a clone of ", name, "\n",
				comment_prefix," the code may change significantly with respect to the original\n",
				comment_prefix," version, especially after program transformations such as dead\n",
				comment_prefix," code elimination and partial evaluation, hence the function may\n",
				comment_prefix," not have the initial behavior, if called under some other context.\n",
				comment_prefix,"\n", NULL));
  text_sentences(t) =
    CONS(SENTENCE, make_sentence(is_sentence_formatted, comments), text_sentences(t));
  add_new_module_from_text(new_name,t,fortran_module_p(cloned),fortran_module_p(cloned)?string_undefined:compilation_unit_of_module(get_current_module_name()));
  free_text(t);

  /* should fix the declarations ... but not the language... */
  entity_initial(new_fun) =
    make_value(is_value_code,
	       make_code(NIL, strdup(""),
			 make_sequence(NIL),
			 NIL,
			 copy_language(l)));

  free_statement(stat);
  free(new_name);
  return new_fun;
}


/********************************************* STORE ALREADY CLONED VERSIONS */

/* already cloned version are kept in a dynamically allocated structure.
 * it is provided with init/close/get/set functions.
 */
typedef struct 
{
    entity the_ref;
    entity the_clone;
    int argn;
    int val;
} clone_t;

static int clones_index = 0 /* next available */, clones_size = 0;
static clone_t * clones = (clone_t*) NULL;

#define INITIALIZED pips_assert("clones initialized", clones && clones_size>0)

static void 
init_clone(void)
{
    pips_assert("clones undefined", clones==NULL && clones_size==0);
    clones_index = 0;
    clones_size = 10;
    clones = (clone_t*) malloc(sizeof(clone_t)*clones_size);
    pips_assert("malloc ok", clones);
}

static void
close_clone(void)
{
    INITIALIZED;
    free(clones), clones = NULL;
    clones_size = 0;
    clones_index = 0;
}

static entity 
get_clone(entity the_ref, int argn, int val)
{
    int i;
    INITIALIZED;
    pips_debug(8, "get %s %d %d\n", entity_name(the_ref), argn, val);
    for (i=0; i<clones_index; i++)
    {
	if (clones[i].the_ref == the_ref &&
	    clones[i].argn == argn &&
	    clones[i].val == val)
	    return clones[i].the_clone;
    }
    return entity_undefined;
}

static void
set_clone(entity the_ref, entity the_clone, int argn, int val)
{
    INITIALIZED;
    pips_debug(8, "put %s %s %d %d\n", 
	       entity_name(the_ref), entity_name(the_clone), argn, val);

    if (clones_index==clones_size)
    {
	clones_size+=10;
	clones = (clone_t*) realloc(clones, sizeof(clone_t)*clones_size);
	pips_assert("realloc ok", clones);
    }

    clones[clones_index].the_ref = the_ref;
    clones[clones_index].the_clone = the_clone;
    clones[clones_index].argn = argn;
    clones[clones_index].val = val;

    clones_index++;
}


/**************************************************** CLONING TRANSFORMATION */

/* static structures used for driving the cloning transformation.
 */
DEFINE_LOCAL_STACK(stmt, statement)

static entity module_to_clone = entity_undefined;
static int argument_to_clone = 0;
static int statement_to_clone = STATEMENT_NUMBER_UNDEFINED;
static entity clonee_to_substitute = entity_undefined;
static bool some_cloning_performed = false;

void clone_error_handler()
{
    error_reset_stmt_stack();
    module_to_clone = entity_undefined;
    argument_to_clone = 0;
    statement_to_clone = STATEMENT_NUMBER_UNDEFINED;
    clonee_to_substitute = entity_undefined;
    some_cloning_performed = false;
}

/* returns if the expression is a constant, maybe thanks to the preconditions.
 */
static bool
this_expression_constant_p(
    expression e, /* expression to be tested */
    int * pval    /* returned integer value if one was found */)
{  
    bool ok = false;
    if (expression_constant_p(e)) 
    {
	pips_debug(7, "constant expression\n");
	*pval = expression_to_int(e);
	ok = true;
    }
    else if (expression_reference_p(e))
    {   
	entity ref;
	statement current;
	Psysteme prec;
	Pbase b;
	Value val;

	ref = reference_variable(expression_reference(e));
	if (entity_integer_scalar_p(ref)) 
	{
	    pips_debug(7, "integer scalar reference\n");
	    
	    /* try with the precondition...
	     */
	    current = stmt_head();
	    prec = sc_dup(predicate_system(transformer_relation(
		load_statement_precondition(current))));
	    b = base_dup(sc_base(prec));
	    vect_erase_var(&b, (Variable) ref);
	    prec = sc_projection_optim_along_vecteur(prec, b);
	    ok = sc_value_of_variable(prec, (Variable) ref, &val);
	    sc_rm(prec), base_rm(b);
	    
	    if (ok) *pval = VALUE_TO_INT(val);
	}
	else pips_debug(7, "not an integer scalar reference\n");
    }
    else if (expression_call_p(e))
    {
	call c = syntax_call(expression_syntax(e));
	entity fun = call_function(c);
	value v = entity_initial(fun);
	
	if (value_symbolic_p(v) &&
	    constant_int_p(symbolic_constant(value_symbolic(v)))) 
	{
	    pips_debug(7, "is an int PARAMETER\n");
	    ok = true;
	    *pval = constant_int(symbolic_constant(value_symbolic(v)));
	}
	else pips_debug(7, "not a symbolic integer constant\n");
    }
    else pips_debug(7, "not a constant nor a reference not a parameter\n");

    pips_debug(5, "ok = %s, val = %d\n", ok? "TRUE": "FALSE", ok? *pval: 0);
    return ok;
}

/* perform a cloning for a given call
 */
static void
do_clone(
    call c,            /* call to be replaced */
    int argn, int val, /* arg number and associated value */
    entity clonee      /* if provided */)
{
    entity cloned = call_function(c);
    pips_debug(3, "%s cloned on argument %d for value %d\n", 
	       entity_name(cloned), argn, val);
    
    /* first check whether the cloning was already performed.
     */
    if (entity_undefined_p(clonee))
    {
	clonee = get_clone(cloned, argn, val);
	if (clonee==entity_undefined)
	{
	    clonee = build_a_clone_for(cloned, argn, val);
	    if (argn!=0) set_clone(cloned, clonee, argn, val);
	}
    }

    some_cloning_performed = true;
    call_function(c) = clonee;
}

static void 
clone_rwt(call c)
{
    expression nth_arg;
    int val = 0;

    if (call_function(c)!=module_to_clone) return;
    pips_debug(3, "considering call to %s\n", entity_name(module_to_clone));

    if (argument_to_clone)
    {
	nth_arg = EXPRESSION(gen_nth(argument_to_clone-1, call_arguments(c)));
	if (this_expression_constant_p(nth_arg, &val)) /* yeah! */
	    do_clone(c, argument_to_clone, val, entity_undefined);
    }
    else if (!undefined_number_p(statement_to_clone) &&
	     statement_to_clone==statement_number(stmt_head()))
    {
	do_clone(c, 0, 0, clonee_to_substitute);
    }
}

/* clone module calls on argument arg in caller.
 * formal parameter of module number argn must be an integer scalar.
 * also used for user-directed cloning or substitution.
 * @return whether okay.
 */
static bool
perform_clone(
    entity module      /* the module being cloned */,
    string caller_name /* the caller of interest */,
    int argn           /* the argument number to be cloned */,
    int number         /* the statement number to clone a call */,
    entity substitute  /* entity to substitute to module */)
{
    statement stat;

    pips_assert("coherent arguments",
		(argn!=0 && undefined_number_p(number) && 
		 entity_undefined_p(substitute)) ||
		(argn==0 && !undefined_number_p(number)));

    pips_debug(2, "cloning %s in %s on %d\n", 
	       entity_local_name(module), caller_name, argn);

    stat = (statement) db_get_memory_resource(DBR_CODE, caller_name, true);

    if (argn!=0)
	set_precondition_map((statement_mapping) 
	    db_get_memory_resource(DBR_PRECONDITIONS, caller_name, true));

    /* init 
     */
    make_stmt_stack();
    module_to_clone = module;
    argument_to_clone = argn;
    statement_to_clone = number;
    clonee_to_substitute = substitute;
    some_cloning_performed = false;

    /* perform cloning
     */
    gen_multi_recurse(stat,
		      statement_domain, stmt_filter, stmt_rewrite,
		      call_domain, gen_true, clone_rwt,
		      NULL);

    /* update CALLEES and CODE if necessary.
     */
    if (some_cloning_performed) 
    {
	DB_PUT_MEMORY_RESOURCE(DBR_CODE, caller_name, stat);
	DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, caller_name,
			       (char *) compute_callees(stat));
    }

    /* close 
     */
    module_to_clone = entity_undefined;
    argument_to_clone = 0;
    statement_to_clone = STATEMENT_NUMBER_UNDEFINED;
    clonee_to_substitute = entity_undefined;
    free_stmt_stack();
    if (argn!=0) reset_precondition_map();

    return some_cloning_performed;
}


/********************************************************************* UTILS */

/* global initializations needed 
 */
static void
set_currents(string name)
{
    entity module;
    statement stat;

    pips_debug(1, "considering module %s\n", name);

    init_clone();

    module = module_name_to_entity(name);
    pips_assert("is a function", type_functional_p(entity_type(module)));
    set_current_module_entity(module);
    
    stat = (statement) db_get_memory_resource(DBR_CODE, name, true);
    set_current_module_statement(stat);
}

/* global resets
 */
static void
reset_currents(const char* name)
{
    close_clone();
    reset_current_module_entity();
    reset_current_module_statement();
    pips_debug(1, "done with module %s\n", name);
}

/* check that caller is indeed a caller of name.
 */
static void
is_a_caller_or_error(
    string name,
    string caller)
{
    callees callers = (callees)db_get_memory_resource(DBR_CALLERS, name, true);
    MAP(STRING, s, 
	if (same_string_p(s, caller)) return, /* ok */
	callees_callees(callers));

    reset_currents(name);
    pips_user_error("%s is not a caller of %s\n", caller, name);    
}

#define invalid_request_result(s) \
	(!(s) || string_undefined_p((s) || (strlen((s))==0))

static string
checked_string_user_request(
    string fmt, string arg,
    string error)
{
    string result = user_request(fmt, arg);
    if (!result || string_undefined_p(result) || (strlen(result)==0))
    {
	reset_currents(entity_local_name(get_current_module_entity()));
	pips_user_error("invalid string for %s\n", error); /* exception */
	return NULL;
    }
    return result;
}

static int
checked_int_user_request(
    string fmt, string arg,
    string error)
{
    int result;
    string si = checked_string_user_request(fmt, arg, error);

    if (sscanf(si, "%d", &result)!=1)
    {
	reset_currents(entity_local_name(get_current_module_entity()));
	pips_user_error("invalid int for %s\n", error); /* throw */
	return 0;
    }
    free(si);
    return result;
}

/******************************************************* PIPSMAKE INTERFACES */

#define ARG_TO_CLONE "TRANSFORMATION_CLONE_ON_ARGUMENT"

/* clone module name, on the argument specified by property
 * int TRANSFORMATION_CLONE_ON_ARGUMENT.
 *
 * clone_on_argument	> CALLERS.callees
 *       		> CALLERS.code
 *       < MODULE.code
 *       < MODULE.callers
 *	 < MODULE.user_file
 *       < CALLERS.callees
 *       < CALLERS.preconditions
 *       < CALLERS.code
 */
bool 
clone_on_argument(string name)
{
    entity module;
    callees callers; /* warf, warf */
    int argn;

    DEBUG_ON;
    set_currents(name);
    module = get_current_module_entity();
    callers = (callees) db_get_memory_resource(DBR_CALLERS, name, true);
    argn = get_int_property(ARG_TO_CLONE);

    if (argn<=0)
    {
	do /* perform a user request to get the argument, 0 to stop */
	{
	    argn = checked_int_user_request("argument of %s to clone", name,
					    "argument number to clone");
	}
	while (argn<0);
    }

    /* check the argument type: must be an integer scalar */
    {
	entity arg = find_ith_parameter(module, argn);
	type t;
	variable v;

	if (entity_undefined_p(arg))
	{
	    reset_currents(name);
	    pips_user_error("%s: no #%d formal\n", name, argn);
	    return false;
	}
	
	t = entity_type(arg);
	pips_assert("arg is a variable", type_variable_p(t));
	v = type_variable(t);

	if (basic_tag(variable_basic(v))!=is_basic_int ||
	    variable_dimensions(v))
	{
	    reset_currents(name);
	    pips_user_error("%s: %d formal not a scalar int\n", name, argn);
	    return false;
	}
    }

    MAP(STRING, caller_name, 
	perform_clone(module, caller_name, argn, 
		      STATEMENT_NUMBER_UNDEFINED, entity_undefined),
	callees_callees(callers));    

    reset_currents(name);
    debug_off();
    return true;
}

/* clone a routine in a caller. the user is requested the caller and
 * ordering to perform the cloning of that instance. can also be used
 * to force a function substitution at a call site.
 *
 * clone 	> CALLERS.code
 * 		> CALLERS.callees
 * 	< MODULE.code
 *      < MODULE.user_file
 *      < CALLERS.code
 *	< CALLERS.callees
 */
static bool
clone_or_clone_substitute(
    string name,
    bool clone_substitute_p)
{
    entity module, substitute;
    string caller;
    int number;
    bool okay;

    DEBUG_ON;
    set_currents(name);
    module = get_current_module_entity();
    
    caller = checked_string_user_request("%s caller to update?", name,
					 "caller to update");

    /* checks whether it is an actual caller. */
    is_a_caller_or_error(name, caller);
	
    number = checked_int_user_request(
	"statement number of %s to clone?", caller,
	"statement number to clone");

    if (clone_substitute_p)
    {
	string substitute_s = 
	    checked_string_user_request("replacement for %s?", name,
					"replacement function");

	/* must be a top-level entity or a C static function */
	substitute = module_name_to_entity(substitute_s);
	if (entity_undefined_p(substitute) || 
	    !type_functional_p(entity_type(substitute)))
	{
	    reset_currents(name);
	    pips_user_error("%s is not an existing function\n", substitute_s);
	    return false;
	}
	free(substitute_s);
    }
    else
	substitute = entity_undefined;

    okay = perform_clone(module, caller, 0, number, substitute);

    if (!okay)
    {
	pips_user_warning("substitution of %s by %s at %s:%d not performed\n",
			  name, 
			  entity_undefined_p(substitute)? 
			      "<none>": entity_local_name(substitute), 
			  caller, number);
    }

    free(caller);
    reset_currents(name);
    debug_off();
    return okay;
}

/* clone name in one of its callers/statement number
 */
bool
clone(string name)
{
    return clone_or_clone_substitute(name, false);
}

/* substitute name in one of its callers/statement number 
 */
bool
clone_substitute(string name)
{
    return clone_or_clone_substitute(name, true);
}

/* use get_current_entity()
 * and get_current_statement()
 * to build a new copy entity
 */
static entity
clone_current_entity()
{
    return build_a_clone_for(get_current_module_entity(), 0,0);
}

/* similar to previous clone and clone_substitute
 * but does not try to make any substitution
 */
#include "preprocessor.h"
bool
clone_only(string mod_name)
{
    /* get the resources */
    statement mod_stmt = (statement)db_get_memory_resource(DBR_CODE, mod_name, true);
    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));
    
    entity cloned_entity = clone_current_entity();
    
    // Used to add the cloned function declaration
    AddEntityToModuleCompilationUnit(cloned_entity, get_current_module_entity());
    
    /* update/release resources */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name,mod_stmt);
    
    reset_current_module_statement();
    reset_current_module_entity();

    return !entity_undefined_p(cloned_entity);
}
