/* HPFC by Fabien Coelho, May 1993 and later...
 *
 * $RCSfile: compile.c,v $ version $Revision$
 * ($Date: 1995/10/10 11:38:03 $, )
 */

#include "defines-local.h"

#include "pipsdbm.h"
#include "resources.h"
#include "effects.h"
#include "semantics.h"
#include "regions.h"
#include "callgraph.h"

extern void AddEntityToDeclarations(entity e, entity f); /* in syntax.h */

#define generate_file_name(filename, prefix, suffix)\
  filename = strdup(concatenate(db_get_current_workspace_directory(),\
				"/", prefix, suffix, NULL));

#define add_warning(filename)\
   safe_system(concatenate("$HPFC_TOOLS/hpfc_add_warning ", filename, NULL));

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
	debug_off();
	fprintf(stderr,"[init_host_and_node_entities]\n old declarations:\n");
	print_text(stderr, text_declaration(current_module));

	fprintf(stderr,"node_module:\n");
	(void) gen_consistent_p(node_module);
	print_text(stderr, text_declaration(node_module));

	fprintf(stderr, "new declarations,\nhost_module:\n");
	(void) gen_consistent_p(host_module);
	print_text(stderr, text_declaration(host_module));

	debug_on("HPFC_DEBUG_LEVEL");
    }
}

void 
put_generated_resources_for_common (entity common)
{
    FILE *host_file, *node_file, *parm_file, *init_file;
    string prefix = entity_local_name(common),
	host_filename, node_filename, parm_filename, init_filename;
    entity
	node_common = load_new_node(common),
	host_common = load_new_host(common);
    
    generate_file_name(host_filename, prefix, "_host.h");
    generate_file_name(node_filename, prefix, "_node.h");
    generate_file_name(parm_filename, prefix, "_parameters.h");
    generate_file_name(init_filename, prefix, "_init.h");

    host_file = (FILE *) safe_fopen(host_filename, "w");
    hpfc_print_common(host_file, host_module, host_common);
    safe_fclose(host_file, host_filename);
    add_warning(host_filename);

    node_file = (FILE *) safe_fopen(node_filename, "w");
    hpfc_print_common(node_file, node_module, node_common);
    safe_fclose(node_file, node_filename);
    add_warning(node_filename);

    parm_file = (FILE *) safe_fopen(parm_filename, "w");
    create_parameters_h(parm_file, common);
    safe_fclose(parm_file, parm_filename);
    add_warning(parm_filename);

    init_file = (FILE *) safe_fopen(init_filename, "w");
    create_init_common_param_for_arrays(init_file, common);
    safe_fclose(init_file, init_filename);
    add_warning(init_filename);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC for common %s\n", 
		entity_name(common));
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
    
    generate_file_name(host_filename, prefix, "_host.f");
    generate_file_name(node_filename, prefix, "_node.f");
    generate_file_name(parm_filename, prefix, "_parameters.h");
    generate_file_name(init_filename, prefix, "_init.h");

    host_file = (FILE *) safe_fopen(host_filename, "w");
    hpfc_print_code(host_file, host_module, host_stat);
    safe_fclose(host_file, host_filename);
    add_warning(host_filename);
    safe_system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
			    host_filename, 
			    " host ", 
			    module_local_name(module),
			    NIL));

    node_file = (FILE *) safe_fopen(node_filename, "w");
    hpfc_print_code(node_file, node_module, node_stat);
    safe_fclose(node_file, node_filename);
    add_warning(node_filename);
    safe_system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
			    node_filename,
			    " node ", 
			    module_local_name(module),
			    NIL));

    parm_file = (FILE *) safe_fopen(parm_filename, "w");
    create_parameters_h(parm_file, module);
    safe_fclose(parm_file, parm_filename);
    add_warning(parm_filename);

    init_file = (FILE *) safe_fopen(init_filename, "w");
    create_init_common_param_for_arrays(init_file, module);
    safe_fclose(init_file, init_filename);
    add_warning(init_filename);

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

    /* free(parm_filename),
    free(init_filename),
    free(host_filename),
    free(node_filename);*/
}

void 
put_generated_resources_for_program (program_name)
string program_name;
{
    FILE *comm_file, *init_file;
    string comm_filename, init_filename, directory_name;

    directory_name = db_get_current_workspace_directory();

    comm_filename = 
	strdup(concatenate(directory_name, "/real_parameters.h", NULL));
    init_filename =
	strdup(concatenate(directory_name, "/hpf_init.h", NULL));

    comm_file = (FILE *) safe_fopen(comm_filename, "w");
    create_common_parameters_h(comm_file);
    safe_fclose(comm_file, comm_filename);
    add_warning(comm_filename);

    init_file = (FILE *) safe_fopen(init_filename, "w");
    create_init_common_param(init_file);
    safe_fclose(init_file, init_filename);
    add_warning(init_filename);
    
    ifdebug(1)
    {
	fprintf(stderr, "Results of HPFC for the program\n");
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(comm_filename);
	hpfc_print_file(init_filename);
    }

    free(comm_filename);
    free(init_filename);
}

/* Compiler call, obsole. left here for allowing linking
 */
void 
hpfcompile (char *module_name)
{
    debug_on("HPFC_DEBUG_LEVEL");
    pips_debug(1, "module: %s\n", module_name);
    pips_internal_error("obsolete\n");
    debug_off();
}

/*   That is all
 */
