/* HPFC by Fabien Coelho, May 1993 and later...
 *
 * $Id$
 * $Log: compile.c,v $
 * Revision 1.68  1997/10/27 16:49:26  coelho
 * db put updated to Src.
 *
 * Revision 1.67  1997/10/27 09:50:27  coelho
 * local basename dropped.
 *
 * Revision 1.66  1997/09/15 19:21:41  coelho
 * fixed for new common prefix.
 *
 * Revision 1.65  1997/08/04 13:53:30  coelho
 * new generic effects.
 *
 * Revision 1.64  1997/07/22 13:21:04  keryell
 * %x -> %p formats.
 *
 * Revision 1.63  1997/04/15 10:57:02  creusil
 * statement_effects used instead of statement mappings. bc.
 *
 * Revision 1.62  1997/03/20 09:49:06  coelho
 * system -> safe_system.
 *
 */

#include "defines-local.h"

#include "pipsdbm.h"
#include "resources.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "callgraph.h"
#include "transformations.h"

extern void AddEntityToDeclarations(entity e, entity f); /* in syntax.h */

#define src(name, suf) \
	strdup(concatenate(WORKSPACE_SRC_SPACE "/", name, suf, 0))

void 
make_host_and_node_modules (entity module)
{
    string name = entity_local_name(module);
    entity host, node;

    if (bound_new_node_p(module))
	return;

    if (entity_main_module_p(module))
    {
	host = make_empty_subroutine(HOST_NAME);
	node = make_empty_subroutine(NODE_NAME);
    }
    else
    {
	string tmp;

	/* HOST and NODE empty routines...
	 */
	tmp = strdup(concatenate(name, "_", HOST_NAME, 0));
	host = make_empty_subroutine(tmp);
	free(tmp);

	tmp = strdup(concatenate(name, "_", NODE_NAME, 0));
	node = make_empty_subroutine(tmp);
	free(tmp);

	/*  Arity and result
	 */
	update_functional_as_model(host, module);
	update_functional_as_model(node, module);

	if (entity_function_p(module))
	{
	    /* then the variable corresponding to the function name
	     * must be created for those new functions. The overloaded
	     * basic is used to be sure that the variable will not be put 
	     * in the declarations by the enforced coherency. 
	     * ??? this issue could be managed by the coherency function ?
	     */
	    string
		var_name = concatenate(name, MODULE_SEP_STRING, name, NULL),
		tmp_name;
	    entity
		var = gen_find_tabulated(var_name, entity_domain), nev;

	    pips_assert("defined", !entity_undefined_p(var));

	    tmp_name = entity_local_name(host);
	    nev = find_or_create_scalar_entity(tmp_name, tmp_name, 
					       is_basic_overloaded);
	    store_new_host_variable(nev, var);

	    tmp_name = entity_local_name(node);
	    nev = find_or_create_scalar_entity(tmp_name, tmp_name, 
					       is_basic_overloaded);
	    store_new_node_variable(nev, var);
	}		
    }

    /*  to allow the update of the call sites.
     */
    store_new_host_variable(host, module);
    store_new_node_variable(node, module);
}

/* kind of a quick hack to remove distributed arguments for the host 
 */
static void 
drop_distributed_arguments(entity module) /* of the host */
{
    type t = entity_type(module);
    functional f;
    list /* of parameter */ le = NIL, lp;
    int len, i, n /* number of next kept parameter */;

    message_assert("functional", type_functional_p(t));
    f = type_functional(t);
    lp = functional_parameters(f);
    len = gen_length(lp);

    pips_debug(8, "considering %d arg(s) of %s\n", len, entity_name(module));

    for (i=1, n=1; i<=len; i++, POP(lp))
    {
	entity ent = find_ith_parameter(module, i);
	
	if (!entity_undefined_p(ent) && !array_distributed_p(ent))
	{
	    le = CONS(PARAMETER, PARAMETER(CAR(lp)), le);
	    pips_debug(8, "keeping %d argument %s\n", i, entity_name(ent));

	    formal_offset(storage_formal(entity_storage(ent))) = n;
	    n++;
	}
	else
	{
	    if (!entity_undefined_p(ent))
		formal_offset(storage_formal(entity_storage(ent))) = INT_MAX;

	    pips_debug(8, "dropping %d argument %s\n", i, 
		       entity_undefined_p(ent)? "undefined": entity_name(ent));
	}
    }

    lp = functional_parameters(f);
    functional_parameters(f) = gen_nreverse(le);
    gen_free_list(lp);
}

static entity 
create_bound_entity(
    entity module,
    entity array,
    bool upper,
    int dim,
    int number)
{
    entity result = argument_bound_entity(module, array, upper, dim);

    free_storage(entity_storage(result));
    entity_storage(result) = 
	make_storage(is_storage_formal, make_formal(module, number));

    pips_assert("formal variable", type_variable_p(entity_type(result)));

    AddEntityToDeclarations(result, module);

    pips_debug(9, "creating %s in %s (arg %d)\n", 
	       entity_name(result), entity_name(module), number);

    return result;
}

static list 
add_one_bound_argument(
    list /* of parameter */ lp,
    entity module,
    entity array,
    bool upper,
    int dim,
    int formal_number)
{
    (void) create_bound_entity(module, array, upper, dim, formal_number);
    lp = CONS(PARAMETER, 
	      make_parameter(make_type(is_type_variable, 
		   make_variable(MakeBasic(is_basic_int), NIL)),
			     make_mode(is_mode_value, UU)), lp);
    return lp;
}

static void
add_bound_arguments(entity module) /* for the node */
{
    type t = entity_type(module);
    functional f;
    list /* of parameter */ le = NIL, lp;
    int len, i, next;

    message_assert("functional", type_functional_p(t));
    f = type_functional(t);
    lp = functional_parameters(f);
    len = gen_length(lp);
    next = len+1;

    for(i=1; i<=len; i++)
    {
	entity arg = find_ith_parameter(module, i),
               old = load_old_node(arg);
	pips_debug(8, "array %s\n", entity_name(old));

	if (array_distributed_p(old))
	{
	    int dim, ndim;
	    ndim = NumberOfDimension(arg);

	    for(dim=1; dim<=ndim; dim++)
	    {	    
		if (ith_dim_overlapable_p(old, dim))
		{
		    le = add_one_bound_argument
			(le, module, arg, FALSE, dim, next++);
		    le = add_one_bound_argument
			(le, module, arg, TRUE, dim, next++);
		}
	    }
	}
    }

    if (le)
    {
	functional_parameters(f) = gen_nconc(lp, le);
    }
}

/* both host and node modules are initialized with the same
 * declarations than the compiled module, but the distributed arrays
 * declarations... which are not declared in the case of the host_module,
 * and the declarations of which are modified in the node_module
 * (call to NewDeclarationsOfDistributedArrays)...
 */
void 
init_host_and_node_entities ()
{
    entity current_module = get_current_module_entity();

    ifdebug(3)
    {
	debug_on("PRETTYPRINT_DEBUG_LEVEL");
	pips_debug(3, "old declarations:\n");
	print_text(stderr, text_declaration(current_module));
	debug_off();
    }

    host_module = load_new_host(current_module);
    node_module = load_new_node(current_module);

    /*  First, the commons are updated
     */
    MAP(ENTITY, e,
     {
	 type t = entity_type(e);

	 if (type_area_p(t) && !SPECIAL_COMMON_P(e))
	 {
	     debug(3, "init_host_and_node_entities",    /* COMMONS */
		   "considering common %s\n", entity_name(e));

	     AddCommonToHostAndNodeModules(e); 
	     add_a_common(e);
	 }
     },
	 entity_declarations(current_module)); 

    /*   Then, the other entities
     */
    MAP(ENTITY, e,
     {
	 type t = entity_type(e);

	 /* parameters are selected. I think they may be either
	  * functional of variable (if declared...) FC 15/09/93
	  */

	 if ((type_variable_p(t)) ||                    /* VARIABLES */
	     ((storage_rom_p(entity_storage(e))) &&
	      (value_symbolic_p(entity_initial(e)))))
	     AddEntityToHostAndNodeModules(e);
	 else
	 if (type_functional_p(t))                      /* PARAMETERS */
	 {
	     AddEntityToDeclarations(e, host_module);   
	     AddEntityToDeclarations(e, node_module);
	 }
     },
	 entity_declarations(current_module));
    
    NewDeclarationsOfDistributedArrays();    

    drop_distributed_arguments(host_module);
    add_bound_arguments(node_module);

    ifdebug(3)
    {
	debug_on("PRETTYPRINT_DEBUG_LEVEL");
	pips_debug(3,"new declarations - node_module:\n");
	(void) gen_consistent_p(node_module);
	print_text(stderr, text_declaration(node_module));

	pips_debug(3, "new declarations - host_module:\n");
	(void) gen_consistent_p(host_module);
	print_text(stderr, text_declaration(host_module));
	debug_off();
    }
}

FILE *
hpfc_fopen(
    string name)
{
    string base = basename(name, NULL);
    FILE *f = (FILE *) safe_fopen(name, "w");
    fprintf(f, "!\n! File %s\n! This file has been automatically generated " 
	    "by the HPF compiler\n!\n", base);
    free(base);
    return f;
}

void
hpfc_fclose(
    FILE *f,
    string name)
{
    string base = basename(name, NULL);
    fprintf(f, "!\n! That is all for %s\n!\n", base);
    free(base);
    safe_fclose(f, name);
}

/* old name of obj while in module now.
 */
static string 
old_name(
    entity module, /* module in which obj appears */
    entity obj)    /* obj */
{
    return module_local_name
        (module==host_module ? load_old_host(obj) : load_old_node(obj));
}

/* to be used by the prettyprinter at the head of a file.
 * inclusion of needed runtime headers.
 */
static string
hpfc_head_hook(
    entity m) /* module */
{
    return strdup(concatenate
        ("      implicit none\n"
	 "      include \"" GLOBAL_PARAMETERS_H "\"\n"
	 "      include \"hpfc_commons.h\"\n"
	 "      include \"hpfc_includes.h\"\n"
	 "      include \"", old_name(m, m), PARM_SUFFIX "\"\n", NULL));
}

/* to be used by the prettyprinter when dealing with a common.
 * inclusion of the parameters and commons...
 */
static string 
hpfc_common_hook(
    entity module,
    entity common)
{
    return strdup(concatenate
        ("      include \"", old_name(module, common), PARM_SUFFIX "\"\n"
	 "      include \"", old_name(module, common),  
	 module==host_module ? HINC_SUFFIX "\"\n" : NINC_SUFFIX "\"\n", NULL));
}

void 
hpfc_print_code(
    FILE* file,
    entity module,
    statement stat)
{
    text t;
    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    set_prettyprinter_head_hook(hpfc_head_hook);
    set_prettyprinter_common_hook(hpfc_common_hook);

    t = text_module(module, stat);
    print_text(file, t); /* t is freed as a side effect... */
    
    reset_prettyprinter_common_hook();
    reset_prettyprinter_head_hook();

    debug_off();
}

#define full_name(dir, name) concatenate(dir, "/", name, NULL)

void 
put_generated_resources_for_common(entity common)
{
    FILE *host_file, *node_file, *parm_file, *init_file;
    string prefix, host_name, node_name, parm_name, init_name, dir_name;
    entity node_common, host_common;

    node_common = load_new_node(common),
    host_common = load_new_host(common);
    prefix = module_local_name(common);
    dir_name = db_get_current_workspace_directory();
    
    host_name = src(prefix, HINC_SUFFIX);
    node_name = src(prefix, NINC_SUFFIX);
    parm_name = src(prefix, PARM_SUFFIX);
    init_name = src(prefix, INIT_SUFFIX);

    host_file = hpfc_fopen(full_name(dir_name, host_name));
    hpfc_print_common(host_file, host_module, host_common);
    hpfc_fclose(host_file, host_name);

    node_file = hpfc_fopen(full_name(dir_name, node_name));
    hpfc_print_common(node_file, node_module, node_common);
    hpfc_fclose(node_file, node_name);

    parm_file = hpfc_fopen(full_name(dir_name, parm_name));
    create_parameters_h(parm_file, common);
    hpfc_fclose(parm_file, parm_name);

    init_file = hpfc_fopen(full_name(dir_name, init_name));
    create_init_common_param_for_arrays(init_file, common);
    hpfc_fclose(init_file, init_name);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC for common %s\n", entity_name(common));
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(parm_name);
	hpfc_print_file(init_name);
	hpfc_print_file(host_name);
	hpfc_print_file(node_name);
    }

    free(parm_name),
    free(init_name),
    free(host_name),
    free(node_name);
}

/* just copied for the host
 */
void
compile_a_special_io_function(entity module)
{
    string prefix, file_name, h_name, dir_name, fs, ft;

    prefix = module_local_name(module);
    file_name = db_get_file_resource(DBR_SOURCE_FILE, prefix, TRUE);
    dir_name = db_get_current_workspace_directory();
    h_name = src(prefix, HOST_SUFFIX);

    fs = strdup(concatenate(dir_name, "/", file_name, 0));
    ft = strdup(concatenate(dir_name, "/", h_name, 0));
    safe_copy(fs, ft);
    free(fs);
    free(ft);

    DB_PUT_FILE_RESOURCE(DBR_HPFC_HOST, prefix, h_name);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_NODE, prefix, NO_FILE);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_RTINIT, prefix, NO_FILE);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_PARAMETERS, prefix, NO_FILE);
}

/* simply copied for both host and node... 
 */
void
compile_a_pure_function(entity module)
{
    string prefix, file_name, hn_name, dir_name, fs, ft;

    pips_debug(1, "compiling pure a function (%s)\n", entity_name(module));

    prefix = module_local_name(module);
    dir_name = db_get_current_workspace_directory();
    file_name = db_get_file_resource(DBR_SOURCE_FILE, prefix, TRUE);
    hn_name = src(prefix, BOTH_SUFFIX);

    fs = strdup(concatenate(dir_name, "/", file_name, 0));
    ft = strdup(concatenate(dir_name, "/", hn_name, 0));
    safe_copy(fs, ft);
    free(fs), free(ft);

    DB_PUT_FILE_RESOURCE(DBR_HPFC_HOST, prefix, hn_name);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_NODE, prefix, strdup(hn_name));
    DB_PUT_FILE_RESOURCE(DBR_HPFC_RTINIT, prefix, NO_FILE);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_PARAMETERS, prefix, NO_FILE);
}

void 
put_generated_resources_for_module(
    statement stat,
    statement host_stat,
    statement node_stat)
{
    FILE *host_file, *node_file, *parm_file, *init_file;
    entity module = get_current_module_entity();
    string prefix, host_name, node_name, parm_name, init_name, dir_name;
    
    prefix = module_local_name(module);
    dir_name = db_get_current_workspace_directory();

    host_name = src(prefix, HOST_SUFFIX);
    host_file = hpfc_fopen(full_name(dir_name, host_name));
    hpfc_print_code(host_file, host_module, host_stat);
    hpfc_fclose(host_file, host_name);

    node_name = src(prefix, NODE_SUFFIX);
    node_file = hpfc_fopen(full_name(dir_name, node_name));
    hpfc_print_code(node_file, node_module, node_stat);
    hpfc_fclose(node_file, node_name);

    parm_name = src(prefix, PARM_SUFFIX);
    parm_file = hpfc_fopen(full_name(dir_name, parm_name));
    create_parameters_h(parm_file, module);
    hpfc_fclose(parm_file, parm_name);

    init_name = src(prefix, INIT_SUFFIX);
    init_file = hpfc_fopen(full_name(dir_name, init_name));
    create_init_common_param_for_arrays(init_file, module);
    hpfc_fclose(init_file, init_name);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC for module %s:\n", 
		module_local_name(module));
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(parm_name);
	hpfc_print_file(init_name);
	hpfc_print_file(host_name);
	hpfc_print_file(node_name);
    }

    DB_PUT_FILE_RESOURCE(DBR_HPFC_PARAMETERS, prefix, parm_name);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_HOST, prefix, host_name);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_NODE, prefix, node_name);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_RTINIT, prefix, init_name);
}

void 
put_generated_resources_for_program (string program_name)
{
    FILE *comm_file, *init_file;
    string comm, init, dir_name;

    dir_name = db_get_current_workspace_directory();

    comm = src(GLOBAL_PARAMETERS_H, "");
    init = src(GLOBAL_INIT_H, "");

    comm_file = hpfc_fopen(full_name(dir_name, comm));
    create_common_parameters_h(comm_file);
    hpfc_fclose(comm_file, comm);

    init_file = hpfc_fopen(full_name(dir_name, init));
    create_init_common_param(init_file);
    hpfc_fclose(init_file, init);
    
    ifdebug(1)
    {
	fprintf(stderr, "Results of HPFC for the program\n");
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(comm);
	hpfc_print_file(init);
    }

    free(comm);
    free(init);
}

/* Compiler call, obsole. left here for allowing linking
 */
void 
hpfcompile (string module_name)
{
    debug_on("HPFC_DEBUG_LEVEL");
    pips_debug(1, "module: %s\n", module_name);
    pips_internal_error("obsolete\n");
    debug_off();
}


/******************************************************* HPFC ATOMIZATION */

/*  drivers for atomize_as_required in transformations
 */
static bool expression_simple_nondist_p(expression e)
{
    reference r;

    if (!expression_reference_p(e)) return(FALSE);

    r = syntax_reference(expression_syntax(e));

    return !array_distributed_p(reference_variable(r)) &&
	    ENDP(reference_indices(r));
}

/* break expression e in reference r if ...
 */
static bool hpfc_decision(reference r, expression e)
{
    return array_distributed_p(reference_variable(r)) &&
	   !expression_integer_constant_p(e) && 
	   !expression_simple_nondist_p(e);
}

entity hpfc_new_variable(module, t)
entity module;
tag t;
{
    return make_new_scalar_variable(module, MakeBasic(t));
}


/********************* EXTRACT NON VARIANT TERMS ON DISTRIBUTED DIMENSIONS */

DEFINE_LOCAL_STACK(c_stmt, statement)

/* true if no written effect on any variables of e in loe 
 */
static bool invariant_expression_p(
    expression e,
    list /* of effect */ loe,
    list /* of entity */ le)
{
    list /* of effect */ l = proper_effects_of_expression(e);

    ifdebug(3) {
	pips_debug(3, "considering expression:");
	print_expression(e);
    }

    MAP(EFFECT, ef1,
	MAP(EFFECT, ef2,
	{
	    variable v = effect_variable(ef1);
	    if ((v==effect_variable(ef2) && effect_write_p(ef2)) ||
		gen_in_list_p(v, le))
	    {
		gen_free_list(l);
		pips_debug(3, "variant\n");
		return FALSE;
	    }
	},
	    loe),
	l);

    gen_free_list(l);
    pips_debug(3, "invariant\n");
    return TRUE;
}

/* substitute all occurences of expression e in statement s by variable v
 */
static entity subs_v;
static expression subs_e;
static Pvecteur subs_pv;

/* returns if vin is in vref
 */
static bool vect_in_p(Pvecteur vin, Pvecteur vref)
{
    Pvecteur v;

    for (v=vin; v; v=v->succ)
    {
	Variable x = vecteur_var(v);
	if (x)
	{
	    if (value_ne(vecteur_val(v),vect_coeff(x, vref)))
		return FALSE;
	}
    }

    return TRUE;
}

/* returns if e is normalized and linear.
 */
bool expression_linear_p(expression e)
{
    normalized n = expression_normalized(e);
    return !normalized_undefined_p(n) && normalized_linear_p(n);
}

static bool expression_flt(expression e)
{
    /* does sg about the linearization of the expression 
     */
    if (subs_pv && expression_linear_p(e))
    {
	normalized n = expression_normalized(e);
	Pvecteur v = normalized_linear(n);
	if (vect_in_p(subs_pv, v))
	{
	    Pvecteur vn = vect_substract(v,subs_pv);
	    vect_add_elem(&vn, (Variable)subs_v, VALUE_ONE);
	    vect_rm(v);
	    normalized_linear_(n) = newgen_Pvecteur(vn);
	}
    }

    if (expression_equal_p(e, subs_e))
    {
	/* ??? memory leak, but how to deal with effect references? */
	expression_syntax(e) = 
	    make_syntax(is_syntax_reference, make_reference(subs_v, NIL));
	free_normalized(expression_normalized(e));
	expression_normalized(e) = 
	    make_normalized(is_normalized_linear,
			    vect_new((Variable)subs_v, VALUE_ONE));
	return FALSE;
    }
    return TRUE;
}
static void substitute_and_create(statement s, entity v, expression e)
{
    instruction i;

    ifdebug(3) {
	pips_debug(3, "variable %s substituted for\n", entity_name(v));
	print_expression(e);
    }

    subs_v = v;
    subs_e = copy_expression(e);

    subs_pv = expression_linear_p(e)?
	vect_dup(normalized_linear(expression_normalized(e))):
	(Pvecteur) NULL;
    
    gen_recurse(s, expression_domain, expression_flt, gen_null);
    
    vect_rm(subs_pv), subs_pv=(Pvecteur)NULL;

    i = loop_to_instruction
	(make_loop(v, 
		   make_range(copy_expression(subs_e),
			      subs_e, /* the copy is reused! */
			      int_to_expression(1)),
		   instruction_to_statement(statement_instruction(s)),
		   entity_empty_label(),
		   make_execution(is_execution_parallel, UU),
		   NIL));

    statement_instruction(s) = i;
}

static bool loop_flt(loop l)
{
    statement s;
    list /* of effect */ loce;
    list /* of entity */ lsubs = NIL;

    if (execution_sequential_p(loop_execution(l)))
	return TRUE;

    s = c_stmt_head();
    loce = effects_effects(load_cumulated_references(s)); 

    MAP(EFFECT, e,
    {
	reference r = effect_reference(e);
	entity v = reference_variable(r);

	if (array_distributed_p(v) && effect_write_p(e))
	{
	    int dim = 0;
	    int p;
	    entity n;

	    pips_debug(3, "considering reference to %s[%d]\n", 
		       entity_name(v), gen_length(reference_indices(r)));

	    MAP(EXPRESSION, x,
	    {
		dim++;
		ifdebug(3) {
		    pips_debug(3, "considering on dim. %d:\n", dim);
		    print_expression(x);
		}
		if (ith_dim_distributed_p(v, dim, &p) &&
		    invariant_expression_p(x, loce, lsubs) &&
		    !expression_integer_constant_p(x))
		{
		    n = hpfc_new_variable(get_current_module_entity(),
					  is_basic_int);
		    substitute_and_create(s, n, x);
		    lsubs = CONS(ENTITY, n, lsubs);
		}
	    },
	        reference_indices(r));
	}
    },
        loce);

    return FALSE;
}

/* transformation: DOALL I,J ... A(I,J,e) -> DOALL E=e,e,1 ,I,J A(I,J,E)
 */
static void 
extract_distributed_non_constant_terms(
    statement s)
{
    DEBUG_STAT(2, "in", s);

    make_c_stmt_stack();
    gen_multi_recurse(s, 
		      statement_domain, c_stmt_filter, c_stmt_rewrite, 
		      loop_domain, loop_flt, gen_null,
		      NULL);
    free_c_stmt_stack();	

    DEBUG_STAT(2, "out", s);
}

/*
 */
void NormalizeCodeForHpfc(statement s)
{
    extract_distributed_non_constant_terms(s);
    normalize_all_expressions_of(s);
    atomize_as_required(s, 
			hpfc_decision,      /* reference test */
			gen_false,          /* function call test */
			ref_to_dist_array_p,/* test condition test */
			hpfc_new_variable);
}

/*************************************************************** COMMONS */

/* To manage the common entities, I decided arbitrarilly that all
 * commons will have to be declared exactly the same way (name and so),
 * so I can safely unify the entities among the modules.
 * The rational for this stupid ugly transformation is that it is much
 * easier to manage the common overlaps for instance if they are
 * simply linked to the same entity.
 * 
 * ??? very ugly, indeed.
 */

GENERIC_CURRENT_MAPPING(update_common, entity, entity)

static void update_common_rewrite(reference r)
{
    entity var = reference_variable(r),
	new_var = load_entity_update_common(var);

    pips_debug(7, "%s to %s\n", entity_name(var), 
	  entity_undefined_p(new_var) ? "undefined" : entity_name(new_var));

    if (!entity_undefined_p(new_var))
	reference_variable(r) = new_var;    
}

static void update_loop_rewrite(l)
loop l;
{
     entity var = loop_index(l),
	new_var = load_entity_update_common(var);

    if (!entity_undefined_p(new_var))
	loop_index(l) = new_var;       
}

static void debug_ref_rwt(reference r)
{
    entity var = reference_variable(r);

    if (entity_in_common_p(var))
	fprintf(stderr, "[debug_ref_rwt] reference to %s\n", 
		entity_name(var));
}

void debug_print_referenced_entities(obj)
gen_chunk *obj;
{
    gen_recurse(obj, reference_domain, gen_true, debug_ref_rwt);
}

void update_common_references_in_obj(obj)
gen_chunk *obj;
{
    gen_multi_recurse(obj, 
		      loop_domain, gen_true, update_loop_rewrite,
		      reference_domain, gen_true, update_common_rewrite,
		      NULL);

    ifdebug(8)
	debug_print_referenced_entities(obj);
}

void update_common_references_in_regions()
{

    STATEMENT_EFFECTS_MAP(stat, effs,
	 {
	     list lef = effects_effects(effs);

	     debug(3, "update_common_references_in_regions",
		   "statement %p (%d effects)\n", 
		   stat, gen_length((list) lef));
	     
	     MAP(EFFECT, e, update_common_rewrite(effect_reference(e)), 
		 (list) lef);
	 },
	     get_rw_effects());
}

void 
NormalizeCommonVariables(
    entity module,
    statement stat)
{
    list ldecl = code_declarations(entity_code(module)), lnewdecl = NIL, ltmp;
    entity common, new_e;

    /* the new entities for the common variables are created and
     * inserted in the common. The declarations are updated.
     */
    MAP(ENTITY, e,
    {
	if (entity_in_common_p(e))
	{
	    common = ram_section(storage_ram(entity_storage(e)));
	    new_e = AddEntityToModule(e, common);
	    ltmp = area_layout(type_area(entity_type(common)));
	    
	    if (gen_find_eq(new_e, ltmp)==entity_undefined)
		gen_insert_after(new_e, e, ltmp);
	    
	    lnewdecl = CONS(ENTITY, new_e, lnewdecl);
	    store_entity_update_common(e, new_e);
	    
	    pips_debug(8, "module %s: %s -> %s\n",
		  entity_name(module), entity_name(e), entity_name(new_e));
	}
	else
	    lnewdecl = CONS(ENTITY, e, lnewdecl);
    },
	ldecl);

    gen_free_list(ldecl);
    code_declarations(entity_code(module)) = lnewdecl;

    /* the references within the program are updated with the new entities
     */
    update_common_references_in_obj(stat);
}

/*   That is all
 */
