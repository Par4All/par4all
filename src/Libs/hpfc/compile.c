/* HPFC by Fabien Coelho, May 1993 and later...
 *
 * $RCSfile: compile.c,v $ version $Revision$
 * ($Date: 1995/12/28 18:21:18 $, )
 */

#include "defines-local.h"

#include "pipsdbm.h"
#include "resources.h"
#include "effects.h"
#include "semantics.h"
#include "regions.h"
#include "callgraph.h"
#include "transformations.h"

extern void AddEntityToDeclarations(entity e, entity f); /* in syntax.h */

static string 
hpfc_local_name (string name, string suffix)
{
    static char buffer[100]; /* ??? should be enough */

    return(sprintf(buffer, "%s_%s", name, suffix));
}

static string 
hpfc_host_local_name (string name)
{
    return(hpfc_local_name(name, HOST_NAME));
}

static string 
hpfc_node_local_name (string name)
{
    return(hpfc_local_name(name, NODE_NAME));
}

void 
make_host_and_node_modules (entity module)
{
    string name = entity_local_name(module);
    entity host, node;

    if (bound_new_node_p(module))
	return;

    if (entity_main_module_p(module))
    {
	host = make_empty_program(HOST_NAME);
	node = make_empty_program(NODE_NAME);
    }
    else 
    {
	host = make_empty_subroutine(hpfc_host_local_name(name));
	node = make_empty_subroutine(hpfc_node_local_name(name));

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
		var = gen_find_tabulated(var_name, entity_domain),
		new = entity_undefined;

	    pips_assert("defined", !entity_undefined_p(var));

	    tmp_name = entity_local_name(host);
	    new = find_or_create_scalar_entity(tmp_name, tmp_name, 
					       is_basic_overloaded);
	    store_new_host_variable(new, var);

	    tmp_name = entity_local_name(node);
	    new = find_or_create_scalar_entity(tmp_name, tmp_name, 
					       is_basic_overloaded);
	    store_new_node_variable(new, var);
	}		
    }

    /*  to allow the update of the call sites.
     */
    store_new_host_variable(host, module);
    store_new_node_variable(node, module);
}

/* init_host_and_node_entities
 *
 * both host and node modules are initialized with the same
 * declarations than the compiled module, but the distributed arrays
 * declarations... which are not declared in the case of the host_module,
 * and the declarations of which are modified in the node_module
 * (call to NewDeclarationsOfDistributedArrays)...
 */
void 
init_host_and_node_entities (void)
{
    entity current_module = get_current_module_entity();

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

	     /* if (gen_find_eq(e, the_commons)!=e)
		 the_commons = CONS(ENTITY, e, the_commons);*/
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

    ifdebug(3)
    {
	debug_on("PRETTYPRINT_DEBUG_LEVEL");

	fprintf(stderr,"[init_host_and_node_entities]\n old declarations:\n");
	print_text(stderr, text_declaration(current_module));

	fprintf(stderr,"node_module:\n");
	(void) gen_consistent_p(node_module);
	print_text(stderr, text_declaration(node_module));

	fprintf(stderr, "new declarations,\nhost_module:\n");
	(void) gen_consistent_p(host_module);
	print_text(stderr, text_declaration(host_module));

	debug_off();
    }
}

FILE *
hpfc_fopen(
    string name)
{
    FILE *f = (FILE *) safe_fopen(name, "w");
    fprintf(f, "c\nc This file has been automatically generated " 
	    "by the hpf compiler\nc\n");
    return f;
}

void
hpfc_fclose(
    FILE *f,
    string name)
{
    fprintf(f, "c\nc That is all\nc\n");
    safe_fclose(f, name);
}

#define generate_file_name(prefix, suffix)\
  strdup(concatenate(db_get_current_workspace_directory(),\
		     "/", prefix, suffix, NULL))

void 
put_generated_resources_for_common (entity common)
{
    FILE *host_file, *node_file, *parm_file, *init_file;
    string prefix = entity_local_name(common),
	host_filename, node_filename, parm_filename, init_filename;
    entity
	node_common = load_new_node(common),
	host_common = load_new_host(common);
    
    host_filename = generate_file_name(prefix, "_host.h");
    node_filename = generate_file_name(prefix, "_node.h");
    parm_filename = generate_file_name(prefix, "_parameters.h");
    init_filename = generate_file_name(prefix, "_init.h");

    host_file = hpfc_fopen(host_filename);
    hpfc_print_common(host_file, host_module, host_common);
    hpfc_fclose(host_file, host_filename);

    node_file = hpfc_fopen(node_filename);
    hpfc_print_common(node_file, node_module, node_common);
    hpfc_fclose(node_file, node_filename);

    parm_file = hpfc_fopen(parm_filename);
    create_parameters_h(parm_file, common);
    hpfc_fclose(parm_file, parm_filename);

    init_file = hpfc_fopen(init_filename);
    create_init_common_param_for_arrays(init_file, common);
    hpfc_fclose(init_file, init_filename);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC for common %s\n", entity_name(common));
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(parm_filename);
	hpfc_print_file(init_filename);
	hpfc_print_file(host_filename);
	hpfc_print_file(node_filename);
    }

    free(parm_filename),
    free(init_filename),
    free(host_filename),
    free(node_filename);
}

void 
put_generated_resources_for_module(stat, host_stat, node_stat)
statement stat, host_stat, node_stat;
{
    FILE *host_file, *node_file, *parm_file, *init_file;
    string
	prefix = module_local_name(get_current_module_entity()),
	host_filename, node_filename, parm_filename, init_filename;
    entity module = get_current_module_entity();
    
    host_filename = generate_file_name(prefix, "_host.f");
    host_file = hpfc_fopen(host_filename);
    hpfc_print_code(host_file, host_module, host_stat);
    hpfc_fclose(host_file, host_filename);

    safe_system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
			    host_filename, " host ", 
			    module_local_name(module), NULL));

    node_filename = generate_file_name(prefix, "_node.f");
    node_file = hpfc_fopen(node_filename);
    hpfc_print_code(node_file, node_module, node_stat);
    hpfc_fclose(node_file, node_filename);

    safe_system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
			    node_filename, " node ", 
			    module_local_name(module), NULL));

    parm_filename = generate_file_name(prefix, "_parameters.h");
    parm_file = hpfc_fopen(parm_filename);
    create_parameters_h(parm_file, module);
    hpfc_fclose(parm_file, parm_filename);

    init_filename = generate_file_name(prefix, "_init.h");
    init_file = hpfc_fopen(init_filename);
    create_init_common_param_for_arrays(init_file, module);
    hpfc_fclose(init_file, init_filename);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC for module %s:\n", 
		module_local_name(module));
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(parm_filename);
	hpfc_print_file(init_filename);
	hpfc_print_file(host_filename);
	hpfc_print_file(node_filename);
    }

    DB_PUT_FILE_RESOURCE(DBR_HPFC_PARAMETERS, strdup(prefix), parm_filename);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_HOST, strdup(prefix), host_filename);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_NODE, strdup(prefix), node_filename);
    DB_PUT_FILE_RESOURCE(DBR_HPFC_RTINIT, strdup(prefix), init_filename);

    /* no! 
    free(parm_filename),
    free(init_filename),
    free(host_filename),
    free(node_filename);
    */
}

void 
put_generated_resources_for_program (program_name)
string program_name;
{
    FILE *comm_file, *init_file;
    string comm, init, directory_name;

    directory_name = db_get_current_workspace_directory();

    comm = strdup(concatenate(directory_name, "/real_parameters.h", NULL));
    init = strdup(concatenate(directory_name, "/hpf_init.h", NULL));

    comm_file = hpfc_fopen(comm);
    create_common_parameters_h(comm_file);
    hpfc_fclose(comm_file, comm);

    init_file = hpfc_fopen(init);
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

void NormalizeCodeForHpfc(statement s)
{
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
    HASH_MAP(stat, lef,
	 {
	     debug(3, "update_common_references_in_regions",
		   "statement 0x%x (%d effects)\n", 
		   (unsigned int) stat, gen_length((list) lef));
	     
	     MAP(EFFECT, e, update_common_rewrite(effect_reference(e)), 
		 (list) lef);
	 },
	     get_local_regions_map());
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
