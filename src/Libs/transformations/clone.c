/* 
 * $Id$
 *
 * Cloning of a subroutine.
 * debug: CLONE_DEBUG_LEVEL
 *
 * $Log: clone.c,v $
 * Revision 1.12  1997/11/04 20:35:43  coelho
 * more comments added to clonee, when cloning on an argument.
 *
 * Revision 1.11  1997/11/04 20:18:04  coelho
 * comment added in front of generated clones. free the text...
 *
 * Revision 1.10  1997/11/04 17:34:05  coelho
 * more comments.
 *
 * Revision 1.9  1997/11/04 13:41:29  coelho
 * version names are formed in exa instead of decimal...
 *
 * Revision 1.8  1997/11/04 13:25:13  coelho
 * more comments.
 *
 * Revision 1.7  1997/11/04 10:36:58  coelho
 * more comments, plus assert to check coherency in perform...
 *
 * Revision 1.6  1997/11/04 10:12:12  coelho
 * clone_substitute interface added...
 *
 * Revision 1.5  1997/11/04 10:06:26  coelho
 * new clone_substitute interface added.
 *
 * Revision 1.4  1997/11/03 18:18:45  coelho
 * cloning seemes ok with both interface... more tests needed however.
 *
 * Revision 1.3  1997/11/03 09:45:33  coelho
 * clone -> clone_on_argument
 * and other clone pass.
 *
 * Revision 1.2  1997/10/31 16:22:10  coelho
 * tmp install for corinne.
 *
 * Revision 1.1  1997/10/31 10:40:46  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"
#include "prettyprint.h"
#include "semantics.h"


#define DEBUG_ON   debug_on("CLONE_DEBUG_LEVEL")
#define ALL_DECLS  "PRETTYPRINT_ALL_DECLARATIONS"
#define STAT_ORDER "PRETTYPRINT_STATEMENT_NUMBER"
#define undefined_number_p(n) ((n)==STATEMENT_NUMBER_UNDEFINED)


/************************************************************ UPDATE CALLEES */

static callees current_callees = callees_undefined;

static void 
callees_rwt(call c)
{
    entity called = call_function(c);
    pips_assert("defined entity", !entity_undefined_p(called));

    if (type_functional_p(entity_type(called)) &&
	storage_rom_p(entity_storage(called)) &&
	(value_code_p(entity_initial(called)) ||
	 value_unknown_p(entity_initial(called)))) 
    {
	string name = entity_local_name(called);
	MAP(STRING, s, 
	    if (same_string_p(name, s)) return, 
	    callees_callees(current_callees));
	callees_callees(current_callees) = 
	    CONS(STRING, strdup(name), callees_callees(current_callees));
    }
}

static callees
compute_callees(statement stat)
{
    callees result;
    current_callees = make_callees(NIL);
    gen_recurse(stat, call_domain, gen_true, callees_rwt);
    result = current_callees;
    current_callees = callees_undefined;
    return result;
}


/************************************************** BUILD THE CLONE VERSIONS */

/* returns an allocated new name for a top-level entity.
 */
static string
build_new_top_level_entity_name(string prefix)
{
    string name = (string) malloc(sizeof(char)*(strlen(prefix)+20)), res;
    int version = 0;

    do { sprintf(name, "%s_%x", prefix, version++); }
    while (local_name_to_top_level_entity(name)!=entity_undefined);

    res = strdup(name); 
    free(name);
    return res;
}

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

	statement_comments(check_arg_value) = strdup(concatenate(
	    "!! PIPS: ", entity_local_name(arg),
	    " is assumed a constant reaching value\n", 0));	    
    }
    else
	check_arg_value = make_continue_statement(entity_undefined);

    stat = make_block_statement(
	CONS(STATEMENT, check_arg_value,
	CONS(STATEMENT, copy_statement(get_current_module_statement()), 
	     NIL)));

    return stat;
}

/* create an clone, and returns the corresponding entity, 
 * which looks like a not yet parsed routine.
 * it puts the initial_file for the routine, and updates its user_file.
 */
static entity
build_a_clone_for(
    entity cloned,
    int argn,
    int val)
{
    string name = entity_local_name(cloned), new_name, comments;
    entity new_fun;
    statement stat;
    type saved_t; 
    storage saved_s;
    value saved_v;
    bool saved_b1, saved_b2;
    text t;

    pips_debug(2, "building a version of %s with arg %d val=%d\n",
	       name, argn, val);
    
    /* builds some kind of module / statement for the clone.
     */
    new_name = build_new_top_level_entity_name(name);
    new_fun = make_empty_function(new_name, copy_type(entity_type(cloned)));

    saved_t = entity_type(new_fun), 
	entity_type(new_fun) = entity_type(cloned);
    saved_s = entity_storage(new_fun),
	entity_storage(new_fun) = entity_storage(cloned);
    saved_v = entity_initial(new_fun),
	entity_initial(new_fun) = entity_initial(cloned);

    saved_b1 = get_bool_property(ALL_DECLS);
    saved_b2 = get_bool_property(STAT_ORDER);
    set_bool_property(ALL_DECLS, TRUE);
    set_bool_property(STAT_ORDER, FALSE);

    stat = build_statement_for_clone(cloned, argn, val);
    t = text_module(new_fun, stat);
    free_statement(stat);

    entity_type(new_fun) = saved_t;
    entity_storage(new_fun) = saved_s;
    entity_initial(new_fun) = saved_v;

    set_bool_property(ALL_DECLS, saved_b1);
    set_bool_property(STAT_ORDER, saved_b2);

    /* add some comments before the code.
     */
    comments = strdup(concatenate(
      "!!\n"
      "!! PIPS: please caution!\n"
      "!!\n"
      "!! this routine has been generated as a clone of ", name, "\n"
      "!! the code may change significantly with respect to the original\n"
      "!! version, especially after program transformations such as dead\n"
      "!! code elimination and partial evaluation, hence the function may\n"
      "!! not have the initial behavior, if called under some other context.\n"
      "!!\n", 0));
    text_sentences(t) = 
	CONS(SENTENCE, make_sentence(is_sentence_formatted, comments),
	     text_sentences(t));

    make_text_resource(new_name, DBR_INITIAL_FILE, ".f_initial", t);
    free_text(t);

    /* give the clonee a user file.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_USER_FILE, new_name, 
	strdup(db_get_memory_resource(DBR_USER_FILE, name, TRUE)));

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
static bool some_cloning_performed;

/* returns if the expression is a constant, maybe thanks to the preconditions.
 */
static bool
this_expression_constant_p(
    expression e, /* expression to be tested */
    int * pval    /* returned integer value if one was found */)
{  
    bool ok = TRUE;
    if (expression_constant_p(e)) 
    {
	*pval = expression_to_int(e);
    }
    else if (expression_reference_p(e))
    {   
	entity ref;
	statement current;
	Psysteme prec;
	Pbase b;
	Value val;

	ref = reference_variable(expression_reference(e));
	if (!entity_integer_scalar_p(ref)) return FALSE;

	/* try with the precondition...
	 */
	current = stmt_head();
	prec = sc_dup(predicate_system(transformer_relation(
	    load_statement_precondition(current))));
	b = base_dup(sc_base(prec));
	vect_erase_var(&b, (Variable) ref);
	prec = sc_projection_optim_along_vecteur(prec, b);
	ok = sc_value_of_variable(prec, (Variable) ref, &val);
	sc_rm(prec);
	base_rm(b);
	
	if (ok) *pval = VALUE_TO_INT(val);
    }

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

    some_cloning_performed = TRUE;
    call_function(c) = clonee;
}

static void 
clone_rwt(call c)
{
    expression nth_arg;
    int val;

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
 */
static void
perform_clone(
    entity module      /* the module being cloned */,
    string caller_name /* the caller of interest */,
    int argn           /* the argument number to be cloned */,
    int number         /* the statement number to clone a call */,
    entity substitute  /* entity to substitute to module */)
{
    entity caller;
    statement stat;

    pips_assert("coherent arguments",
		(argn!=0 && undefined_number_p(number) && 
		 entity_undefined_p(substitute)) ||
		(argn==0 && !undefined_number_p(number)));

    pips_debug(2, "cloning %s in %s on %d\n", 
	       entity_local_name(module), caller_name, argn);

    caller = local_name_to_top_level_entity(caller_name);
    stat = (statement) db_get_memory_resource(DBR_CODE, caller_name, TRUE);

    if (argn!=0)
	set_precondition_map((statement_mapping) 
	    db_get_memory_resource(DBR_PRECONDITIONS, caller_name, TRUE));

    /* init 
     */
    make_stmt_stack();
    module_to_clone = module;
    argument_to_clone = argn;
    statement_to_clone = number;
    clonee_to_substitute = substitute;
    some_cloning_performed = FALSE;

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

    module = local_name_to_top_level_entity(name);
    pips_assert("is a function", type_functional_p(entity_type(module)));
    set_current_module_entity(module);
    
    stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
    set_current_module_statement(stat);
}

/* global resets
 */
static void
reset_currents(string name)
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
    callees callers = (callees)db_get_memory_resource(DBR_CALLERS, name, TRUE);
    MAP(STRING, s, 
	if (same_string_p(s, caller)) return, /* ok */
	callees_callees(callers));
    pips_user_error("%s is not a caller of %s\n", caller, name);    
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
    callers = (callees) db_get_memory_resource(DBR_CALLERS, name, TRUE);
    argn = get_int_property(ARG_TO_CLONE);

    if (argn<=0)
    {
	do /* perform a user request to get the argument, 0 to stop */
	{
	    string args = user_request("argument of %s to clone", name);
	    argn = atoi(args); 
	    free(args);
	}
	while (argn<0);
    }

    /* check the argument type: must be an integer scalar */
    {
	entity arg = find_ith_parameter(module, argn);
	type t;
	variable v;

	if (entity_undefined_p(arg))
	    pips_user_error("%s: no #%d formal\n", name, argn);
	
	t = entity_type(arg);
	pips_assert("arg is a variable", type_variable_p(t));
	v = type_variable(t);

	if (basic_tag(variable_basic(v))!=is_basic_int ||
	    variable_dimensions(v))
	    pips_user_error("%s: %d formal not a scalar int\n", name, argn);
    }

    MAP(STRING, caller_name, 
	perform_clone(module, caller_name, argn, 
		      STATEMENT_NUMBER_UNDEFINED, entity_undefined),
	callees_callees(callers));    

    reset_currents(name);
    debug_off();
    return TRUE;
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
    string caller, number_s;
    int number;

    DEBUG_ON;
    set_currents(name);
    module = get_current_module_entity();
    
    caller = user_request("%s caller to update?", name);
    is_a_caller_or_error(name, caller);
	
    number_s = user_request("statement number of %s to clone?", caller);
    number = atoi(number_s);
    free(number_s);

    if (clone_substitute_p)
    {
	string substitute_s = user_request("replacement for %s?", name);
	substitute = local_name_to_top_level_entity(substitute_s);
	if (entity_undefined_p(substitute) || 
	    !type_functional_p(entity_type(substitute)))
	    pips_user_error("%s is not an existing function\n", substitute_s);
	free(substitute_s);
    }
    else
	substitute = entity_undefined;

    perform_clone(module, caller, 0, number, substitute);

    free(caller);
    reset_currents(name);
    debug_off();
    return TRUE;
}

/* clone name in one of its callers/statement number
 */
bool
clone(string name)
{
    return clone_or_clone_substitute(name, FALSE);
}

/* substitute name in one of its callers/statement number 
 */
bool
clone_substitute(string name)
{
    return clone_or_clone_substitute(name, TRUE);
}
