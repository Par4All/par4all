/*
 * HPFC
 * 
 * Fabien Coelho, May 1993
 *
 * SCCS Stuff:
 * $RCSfile: compile.c,v $ ($Date: 1994/12/27 08:53:16 $) version $Revision$, got on %D%, %T%
 * %A%
 */

/*
 * included files, from C libraries, newgen and pips libraries.
 */

#include <stdio.h>
#include <string.h>

extern int fprintf();
extern int vfprintf();
extern int system();

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "hpf.h"
#include "hpf_private.h"

#include "misc.h"
#include "ri-util.h"
#include "properties.h"
#include "pipsdbm.h"
#include "resources.h"
#include "effects.h"
#include "semantics.h"
#include "regions.h"
#include "callgraph.h"
#include "hpfc.h"
#include "defines-local.h"

/* external functions */
extern char *getenv();
extern void AddEntityToDeclarations(entity e, entity f); /* in syntax.h */

/*  GLOBAL VARIABLES
 */
static list the_callees = NIL;

/*    global list of encountered commons and modules
 */
static list the_commons = NIL;
static list the_modules = NIL;

GENERIC_CURRENT_MAPPING(hpfc_already_compiled, bool, entity);

static string 
hpfc_find_a_not_compiled_module (void)
{
    entity module;
    
    MAPL(ce,
     {
	 module = ENTITY(CAR(ce));
	 
	 if (entity_hpfc_already_compiled_undefined_p(module))
	     return(module_local_name(module));
     },
	 the_modules);

    return(string_undefined);
}

#define generate_file_name(filename, prefix, suffix)\
  filename = strdup(concatenate(db_get_current_program_directory(),\
				"/", prefix, suffix, NULL));

#define add_warning(filename)\
  system(concatenate("$HPFC_TOOLS/hpfc_add_warning ", filename, NULL));


void 
reset_resources_for_module (string module_name)
{
    debug(5, "reset_resources_for_module",
	  "resetting resources of module %s\n", module_name);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_effects_map();
    reset_proper_effects_map();
    reset_local_regions_map();
    reset_precondition_map();

    free_only_io_map();
    free_postcondition_map();
    free_host_gotos_map();
    free_node_gotos_map();

    gen_free_list(the_callees), the_callees = NIL;
}
		     
void 
set_resources_for_module (string module_name)
{
    entity
	module = local_name_to_top_level_entity(module_name),
	stop = entity_undefined;

    db_set_current_module_name(module_name);
    set_current_module_entity(module);
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, FALSE));

    /*   PRECONDITIONS
     */
    set_precondition_map
	((statement_mapping)
	 db_get_memory_resource(DBR_PRECONDITIONS, module_name, FALSE));

    /*   POSTCONDITIONS
     */
    set_postcondition_map
	(compute_postcondition(get_current_module_statement(),
			       MAKE_STATEMENT_MAPPING(),
			       get_precondition_map()));

    /*   REGIONS
     */
    set_local_regions_map
	(effectsmap_to_listmap((statement_mapping)
	 db_get_memory_resource(DBR_REGIONS, module_name, FALSE)));
    
    /*
     * Initialize mappings
     */
    only_io_mapping_initialize(get_current_module_statement());

    make_host_gotos_map();
    make_node_gotos_map();

    /*   CALLEES
     */
    the_callees = entity_to_callees(module);

    make_host_and_node_modules(module);

    MAPL(ce,
     {
	 entity callee = ENTITY(CAR(ce));

	 assert(entity_module_p(callee));

	 if (!value_intrinsic_p(entity_initial(callee)) &&
	     !hpfc_entity_reduction_p(callee) &&
	     !hpfc_intrinsic_like_function(callee))
	 {
	     debug(4, "set_resources_for_module", 
		   "callee %s to be {host,node}ify\n", entity_name(callee));

	     make_host_and_node_modules(callee);
	     if (gen_find_eq(callee, the_modules)!=callee)
		 the_modules = CONS(ENTITY, callee, the_modules);
	 }
     },
	 the_callees);

    hpfc_init_run_time_entities();

    /*   STOP is to be translated into hpfc_{host,node}_end
     */
    stop = local_name_to_top_level_entity(STOP_FUNCTION_NAME);
    store_new_host_variable(hpfc_name_to_entity(HOST_END), stop);
    store_new_node_variable(hpfc_name_to_entity(NODE_END), stop);

    debug(1, "set_resources_for_module", 
	  "building %s.hpf in %s\n",
	  module_name,
	  db_get_current_program_directory());

    system(concatenate("$HPFC_TOOLS/hpfc_filter < ",
		       db_get_file_resource(DBR_SOURCE_FILE, module_name, TRUE),
		       " > ",
		       db_get_current_program_directory(),
		       "/",
		       module_name,
		       ".hpf",
		       NIL));
}

void 
put_generated_resources_for_common (entity common)
{
    FILE 
	*host_file,
	*node_file,
	*parm_file,
	*init_file;
    string
	prefix = entity_local_name(common),
	host_filename,
	node_filename,
	parm_filename,
	init_filename;
    entity
	node_common = load_entity_node_new(common),
	host_common = load_entity_host_new(common);
    
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

	hpfc_print_file(stderr, parm_filename);
	hpfc_print_file(stderr, init_filename);
	hpfc_print_file(stderr, host_filename);
	hpfc_print_file(stderr, node_filename);
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
    FILE 
	*host_file,
	*node_file,
	*parm_file,
	*init_file;
    string
	prefix = db_get_current_module_name(),
	host_filename,
	node_filename,
	parm_filename,
	init_filename;
    entity
	module = get_current_module_entity();
    
    generate_file_name(host_filename, prefix, "_host.f");
    generate_file_name(node_filename, prefix, "_node.f");
    generate_file_name(parm_filename, prefix, "_parameters.h");
    generate_file_name(init_filename, prefix, "_init.h");

    host_file = (FILE *) safe_fopen(host_filename, "w");
    hpfc_print_code(host_file, host_module, host_stat);
    safe_fclose(host_file, host_filename);
    add_warning(host_filename);
    system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
		       host_filename, 
		       " host ", 
		       module_local_name(module),
		       NIL));

    node_file = (FILE *) safe_fopen(node_filename, "w");
    hpfc_print_code(node_file, node_module, node_stat);
    safe_fclose(node_file, node_filename);
    add_warning(node_filename);
    system(concatenate("$HPFC_TOOLS/hpfc_add_includes ", 
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

	hpfc_print_file(stderr, parm_filename);
	hpfc_print_file(stderr, init_filename);
	hpfc_print_file(stderr, host_filename);
	hpfc_print_file(stderr, node_filename);
    }

    free(parm_filename),
    free(init_filename),
    free(host_filename),
    free(node_filename);
}

void 
put_generated_resources_for_program (program_name)
string program_name;
{
    FILE
	*comm_file,
	*init_file;
    string
	comm_filename,
	init_filename;

    comm_filename = 
	strdup(concatenate(db_get_current_program_directory(),
			   "/real_parameters.h",
			   NULL));
    init_filename =
	strdup(concatenate(db_get_current_program_directory(),
			   "/hpf_init.h",
			   NULL));

    comm_file = (FILE *) safe_fopen(comm_filename, "w");
    create_common_parameters_h(comm_file);
    safe_fclose(comm_file, comm_filename);
    add_warning(comm_filename);

    init_file = (FILE *) safe_fopen(init_filename, "w");
    create_init_common_param(init_file);
    safe_fclose(init_file, init_filename);
    add_warning(init_filename);

    system(concatenate("$HPFC_TOOLS/hpfc_generate_init -n ",
		       program_name,
		       " ",
		       db_get_current_program_directory(),
		       NULL));
    
    ifdebug(1)
    {
	fprintf(stderr, "Results of HPFC for the program\n");
	fprintf(stderr, "-----------------\n");

	hpfc_print_file(stderr, comm_filename);
	hpfc_print_file(stderr, init_filename);
    }

    free(comm_filename);
    free(init_filename);
}

void 
init_hpfc_for_program (void)
{
    /* HPFC-PACKAGE is used to put dummy variables in, and other things
     */
    (void) make_empty_program(HPFC_PACKAGE);

    the_commons = NIL;
    the_modules = NIL;
    reset_hpf_object_lists();

    hpfc_init_unique_numbers();

    init_hpf_number_management();
    init_overlap_management();

    make_align_map();       /* ??? memory leak */
    make_distribute_map();  /* ??? memory leak */
    make_host_node_maps();
    make_hpfc_already_compiled_map();
    make_hpfc_current_mappings();
    make_new_declaration_map();
    make_referenced_variables_map();
    make_update_common_map();
}

void 
close_hpfc_for_program (void)
{
    gen_free_list(the_commons), the_commons = NIL;
    gen_free_list(the_modules), the_modules = NIL;
    free_hpf_object_lists();

    close_hpf_number_management();
    close_overlap_management();

    free_align_map();
    free_distribute_map();
    free_host_node_maps();
    free_hpfc_already_compiled_map();
    free_hpfc_current_mappings();
    free_new_declaration_map();
    free_referenced_variables_map(); 
    free_update_common_map();
}

void 
hpfcompile_common (string common_name)
{
    entity 
	common = local_name_to_top_level_entity(common_name);

    declaration_with_overlaps_for_module(common);
    clean_common_declaration(load_entity_host_new(common));
    put_generated_resources_for_common(common);

    store_entity_hpfc_already_compiled(common, TRUE);
}

void 
hpfcompile_module (string module_name)
{
    entity 
	module = entity_undefined;
    statement   
	module_stat,
	host_stat,
	node_stat;

    set_resources_for_module(module_name);

    module = get_current_module_entity();
    module_stat = get_current_module_statement();

    /* what is to be done 
     * filter the source for the directives
     * read them
     * get the code,
     * hpf_normalize the code,
     * get the effects on the normalized code, !!!!
     * touch the declarations,
     * initiate both host and node,
     * generate the run-time data structure,
     * compile,
     * then put in the db the results of the compiler,
     * and find a way to print it!
     */

    NormalizeCommonVariables(module, module_stat);
    ReadHpfDir(module_name);
    NormalizeHpfDeclarations();
    NormalizeCodeForHpfc(module_stat);

    /* put here because the current module is updated with some external
     * declarations 
     */
    init_host_and_node_entities(); 
    
    host_stat = statement_undefined, node_stat = statement_undefined;

    hpfcompiler(module_stat, &host_stat, &node_stat);
    
    if (entity_main_module_p(module))
	add_pvm_init_and_end(&host_stat, &node_stat);

    declaration_with_overlaps_for_module(module);

    update_object_for_module(node_stat, node_module);
    update_object_for_module(entity_code(node_module), node_module);
    insure_declaration_coherency(node_module, node_stat);

    update_object_for_module(host_stat, host_module);
    update_object_for_module(entity_code(host_module), host_module);
    insure_declaration_coherency(host_module, host_stat);

    put_generated_resources_for_module(module_stat, host_stat, node_stat);
    
    reset_resources_for_module(module_name);

    /*   Now the module is compiled
     */
    store_entity_hpfc_already_compiled(module, TRUE);
}

/*
 * hpfcompile
 *
 * Compiler call
 */
void 
hpfcompile (char *module_name)
{
    string name = NULL;

    debug_on("HPFC_DEBUG_LEVEL");

    debug(1, "hpfcompile", "module: %s\n", module_name);

    set_bool_property("PRETTYPRINT_COMMONS", FALSE); 
    set_bool_property("PRETTYPRINT_HPFC", TRUE);

    init_hpfc_for_program();

    hpfcompile_module(module_name);

    while(!string_undefined_p(name=hpfc_find_a_not_compiled_module()))
    {
	db_set_current_module_name(name);
	hpfcompile_module(name);
    }

    /*    COMMONS
     */
    set_bool_property("PRETTYPRINT_COMMONS", TRUE); 
    db_set_current_module_name(module_name);

    MAPL(ce,
     {
	 hpfcompile_common(entity_local_name(ENTITY(CAR(ce))));
     },
	 the_commons);
    
    put_generated_resources_for_program(module_name);

    close_hpfc_for_program();

    debug_off();
}

/*
 * ReadHpfDir
 */
void 
ReadHpfDir (string module_name)
{
    debug(8,"ReadHpfDir", "module: %s\n", module_name);
    
    /* filter 
     */
    hpfcparser(module_name);
    
}

static string 
hpfc_local_name (string name, string suffix)
{
    static char buffer[100]; /* should be enough */

    return(sprintf(buffer, "%s_%s", name, suffix));
}

string 
hpfc_host_local_name (string name)
{
    return(hpfc_local_name(name, HOST_NAME));
}

string 
hpfc_node_local_name (string name)
{
    return(hpfc_local_name(name, NODE_NAME));
}

void 
make_host_and_node_modules (entity module)
{
    string
	name = entity_local_name(module);
    entity
	host = entity_undefined,
	node = entity_undefined;

    if (!entity_node_new_undefined_p(module))
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

	    assert(!entity_undefined_p(var));

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

/*
 * init_host_and_node_entities
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
    entity
	current_module = get_current_module_entity();

    host_module = load_entity_host_new(current_module);
    node_module = load_entity_node_new(current_module);

    /*  First, the commons are updated
     */
    MAPL(ce,
     {
	 entity 
	     e = ENTITY(CAR(ce));
	 type
	     t = entity_type(e);

	 if (type_area_p(t) && !SPECIAL_COMMON_P(e))
	 {
	     debug(3, "init_host_and_node_entities",    /* COMMONS */
		   "considering common %s\n", entity_name(e));

	     AddCommonToHostAndNodeModules(e); 

	     if (gen_find_eq(e, the_commons)!=e)
		 the_commons = CONS(ENTITY, e, the_commons);
	 }
     },
	 entity_declarations(current_module)); 

    /*   Then, the other entities
     */
    MAPL(ce,
     {
	 entity 
	     e = ENTITY(CAR(ce));
	 type
	     t = entity_type(e);

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
