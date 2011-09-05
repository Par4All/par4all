#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "genC.h"
#include "newgen_set.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

#include "callgraph.h"

#include "misc.h"

static set ops_set = NULL;
static set suffixes_set = NULL;
static const char* prefix = NULL;

//Helper macro for tiedous tasks.
#define __KV(key, value) if(strcmp( key ,s) == 0) return value;
#define __MKV(key, value) __KV(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING key , value)

//Returns the name of an operation.
//NULL if not found.
static const char* opname(const char* s)
{
	// /!\ When adding operators to this list,
	// don’t forget to update the corresponding table in the
	// Rename Operator section of pipsmake-rc.tex
	 
	__MKV(POST_INCREMENT_OPERATOR_NAME, "post_inc")
	__MKV(PRE_INCREMENT_OPERATOR_NAME, "inc_pre")

	__MKV(POST_DECREMENT_OPERATOR_NAME, "post_dec")
	__MKV(PRE_DECREMENT_OPERATOR_NAME, "dec_pre")

	__MKV(PLUS_C_OPERATOR_NAME, "plus")
	__MKV(PLUS_OPERATOR_NAME, "plus")
	__MKV(UNARY_PLUS_OPERATOR_NAME, "un_plus")
	__MKV(MINUS_C_OPERATOR_NAME, "minus")
	__MKV(MINUS_OPERATOR_NAME, "minus")
	__MKV(UNARY_MINUS_OPERATOR_NAME, "un_minus")

	__MKV(MULTIPLY_OPERATOR_NAME, "mul")
	__MKV(DIVIDE_OPERATOR_NAME, "div")
	__MKV(MODULO_OPERATOR_NAME, "mod")

	__MKV(ASSIGN_OPERATOR_NAME, "assign")
	__MKV(MULTIPLY_UPDATE_OPERATOR_NAME, "mul_up")
	__MKV(DIVIDE_UPDATE_OPERATOR_NAME, "div_up")
	__MKV(MODULO_UPDATE_OPERATOR_NAME, "mod_up")
	__MKV(PLUS_UPDATE_OPERATOR_NAME, "plus_up")
	__MKV(MINUS_UPDATE_OPERATOR_NAME, "minus_up")

	__MKV(C_LESS_OR_EQUAL_OPERATOR_NAME, "leq")
	__MKV(C_LESS_THAN_OPERATOR_NAME, "lt")
	__MKV(C_GREATER_OR_EQUAL_OPERATOR_NAME, "geq")
	__MKV(C_GREATER_THAN_OPERATOR_NAME, "gt")
	__MKV(C_EQUAL_OPERATOR_NAME, "eq")
	__MKV(C_NON_EQUAL_OPERATOR_NAME, "neq")
	return NULL;
}

static bool take_lvalue(const char* s)
{
	__MKV(ASSIGN_OPERATOR_NAME, true)
	__MKV(MULTIPLY_UPDATE_OPERATOR_NAME, true)
	__MKV(DIVIDE_UPDATE_OPERATOR_NAME, true)
	__MKV(MODULO_UPDATE_OPERATOR_NAME, true)
	__MKV(PLUS_UPDATE_OPERATOR_NAME, true)
	__MKV(MINUS_UPDATE_OPERATOR_NAME, true)
	return false;
}

//Returns the short suffix associated with a type.
//NULL if not found.
static const char* typesuffix(const char* s)
{
	// /!\ When adding types to this list,
	// don’t forget to update the corresponding table in the
	// Rename Operator section of pipsmake-rc.tex
	
	__KV("char","c")
	__KV("short","s")
	__KV("int","i")
	__KV("long","l")
	__KV("float","f")
	__KV("double","d")
	__KV("_Bool","b")
	__KV("_Complex","C")
	__KV("_Imaginary","I")
	return NULL;
}

static
void rename_op(call c)
{
    //TODO: Unary operators

    //Retrieve operator name
    entity f = call_function(c);
    string fname = entity_name(f);

    const char* name = opname(fname);

    if(!name || !set_belong_p(ops_set, name))
        return; //Not an operator, skip it !

    //Retrieve arguments type.
    //All arguments should have the same type

    list args = call_arguments(c);

    string tname = NULL; //Arguments type

    FOREACH(EXPRESSION, arg, args)
    {
        type t = expression_to_type(arg); //Get argument type
        
        if(type_variable_p(t))
        {
            string n = basic_to_string(variable_basic(type_variable(t)));
            if(tname && !same_string_p(tname,n))
            {
                //Arguments have not the same type
                //Skip it
                free_type(t); free(n);
                return;
            }
            free(tname);
            tname = n;
        }
        else
        {
        	//Not a basic type, maybe a compound type
        	//Skip it
        	free_type(t);
        	return;
        }
        free_type(t);
    }

    if(!tname)
	    //Nullary function, quite strange for an operator
	    return;

    //Try to find the suffix
    const char* suffix = typesuffix(tname);
    if(!suffix || !set_belong_p(suffixes_set, suffix))
    {
        free(tname);
        return; //Unknow suffix, is it really a basic type ?
    }

    //Now try to construct a complete function name for the operator
    //Function name look like TOP-LEVEL:<prefix><op name><type suffix>

    string fullname = malloc(1 + strlen(prefix)  + strlen(name) + strlen(suffix));
    strcat(strcat(strcpy(fullname, prefix), name), suffix);

    //Find the associated function in the entity list
    entity newe = FindEntity(TOP_LEVEL_MODULE_NAME, fullname);

    if(newe == entity_undefined)
    {
	    //Missing function ?
	    pips_user_warning("No function %s found, skipping operator %s\n", fullname, fname);
    	free(fullname);
    	free(tname);
	    return;
    }

    if(take_lvalue(fname))
    {
	    //Takes a lvalue as first argument
	    expression farg = EXPRESSION(CAR(args));
	    list l = CONS(EXPRESSION, copy_expression(farg), NIL);
        entity ifunc = entity_intrinsic(ADDRESS_OF_OPERATOR_NAME);
	    update_expression_syntax(farg, make_syntax_call(make_call(ifunc, l)));
    }

    call_function(c) = newe; //Replace the operator by the new function

    free(fullname);
    free(tname);
}

static
void rw_loop(statement sl)
{
    if(statement_loop_p(sl))
        do_loop_to_for_loop(sl);
}

static 
set make_string_set_from_prop(const char* prop)
{
    set s = set_make(set_string);
    list l = strsplit(prop," ");
    set_append_list(s, l);
    //list_free(l);
    return s;
}


/* A short pass that replace operators by function calls
 */
bool rename_operator(const char* module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
    
    /* some properties */
    const char* ops_string = get_string_property("RENAME_OPERATOR_OPS");
    ops_set = make_string_set_from_prop(ops_string);

    const char* suffixes_string = get_string_property("RENAME_OPERATOR_SUFFIXES");
    suffixes_set = make_string_set_from_prop(suffixes_string);
    
    prefix = get_string_property("RENAME_OPERATOR_PREFIX");
    
    /* search */
    if(get_bool_property("RENAME_OPERATOR_REWRITE_DO_LOOP_RANGE"))
        gen_recurse(get_current_module_statement(), statement_domain, gen_true, rw_loop);
    gen_recurse(get_current_module_statement(), call_domain, gen_true, rename_op);

    /* free properties */
    set_free(ops_set);
    ops_set = NULL;

    set_free(suffixes_set);
    suffixes_set = NULL;

    prefix = NULL;

    /* update ressources */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(get_current_module_statement()));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
	return true;
}
