/* HPFC module by Fabien COELHO
 *
 * $RCSfile: hpfc.c,v $ ($Date: 1995/04/28 11:48:14 $, )
 * version $Revision$
 */
 
#include "defines-local.h"

#include "regions.h"
#include "semantics.h"
#include "effects.h"
#include "resources.h"
#include "pipsdbm.h"
#include "control.h"

#define NO_FILE "no file name"

/*---------------------------------------------------------------------
 *
 *  COMMONS
 *
 */

GENERIC_STATIC_STATUS(/**/, the_commons, list, NIL, gen_free_list)

void add_a_common(c)
entity c;
{
    the_commons = gen_once(c, the_commons);
}

static void compile_common(c)
entity c;
{
    declaration_with_overlaps_for_module(c);
    clean_common_declaration(load_new_host(c));
    put_generated_resources_for_common(c);
}

GENERIC_STATIC_STATUS(/**/, the_pures, list, NIL, gen_free_list)

void add_a_pure(f)
entity f;
{
    the_pures = gen_once(f, the_pures);
}

/* ??? some intrinsics should also be considered as pure. all of them ?
 */
bool hpf_pure_p(f)
entity f;
{
    return(gen_in_list_p(f, the_pures));
}

/*---------------------------------------------------------------------
 *
 *  COMPILER STATUS MANAGEMENT
 */
/* initialization of data that belongs to the hpf compiler status
 */
static void init_hpfc_status()
{
    init_entity_status();
    init_data_status();
    init_hpf_number_status();
    init_overlap_status();
    init_the_commons();
    init_dynamic_status();
    init_the_pures();
}

static void reset_hpfc_status()
{
    reset_entity_status();
    reset_data_status();
    reset_hpf_number_status();
    reset_overlap_status();
    reset_the_commons();
    reset_dynamic_status();
    reset_the_pures();
}

static void save_hpfc_status() /* GET them */
{
    string name = db_get_current_program_name();
    hpfc_status s = 
	make_hpfc_status(get_overlap_status(),
			 get_data_status(),
			 get_hpf_number_status(),
			 get_entity_status(),
			 get_the_commons(),
			 get_dynamic_status(),
			 get_the_pures());    

    DB_PUT_MEMORY_RESOURCE(DBR_HPFC_STATUS, strdup(name), s);

    reset_hpfc_status(); /* cleaned! */
}

static void load_hpfc_status() /* SET them */
{
    string name = db_get_current_program_name();
    hpfc_status	s = (hpfc_status) 
	db_get_resource(DBR_HPFC_STATUS, name, TRUE);

    set_entity_status(hpfc_status_entity_status(s));
    set_overlap_status(hpfc_status_overlapsmap(s));
    set_data_status(hpfc_status_data_status(s));
    set_hpf_number_status(hpfc_status_numbers_status(s));
    set_the_commons(hpfc_status_commons(s));
    set_dynamic_status(hpfc_status_dynamic_status(s));
    set_the_pures(hpfc_status_pures(s));
}

static void close_hpfc_status()
{
    close_entity_status();
    close_data_status();
    close_hpf_number_status();
    close_overlap_status();
    close_the_commons();
    close_dynamic_status();
    close_the_pures();

    reset_hpfc_status();
}

/*---------------------------------------------------------------------
 *
 *  COMPILATION
 *
 */

static void set_resources_for_module(module)
entity module;
{
    string module_name = module_local_name(module);
    entity stop;

    /*   STATEMENT
     */
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
    
    /*   ONLY I/O
     */
    only_io_mapping_initialize(get_current_module_statement());
    
    reset_unique_numbers();

    /*   OTHERS
     */
    make_hpfc_current_mappings();

    /*  next in hpfc_init ???
     */
    hpfc_init_run_time_entities();

    /*   STOP is to be translated into hpfc_{host,node}_end
     */
    stop = local_name_to_top_level_entity(STOP_FUNCTION_NAME);
    store_new_host_variable(hpfc_name_to_entity(HOST_END), stop);
    store_new_node_variable(hpfc_name_to_entity(NODE_END), stop);

}

static void 
reset_resources_for_module()
{
    reset_current_module_statement();
    reset_local_regions_map();
    reset_precondition_map();

    free_only_io_map();
    free_postcondition_map();

    free_hpfc_current_mappings();
}

static void compile_module(module)
entity module;
{
    statement s, 
        host_stat = statement_undefined, 
        node_stat = statement_undefined;

    /*   INIT
     */
    set_resources_for_module(module);
    s = get_current_module_statement();
    make_host_and_node_modules(module);

    /*   NORMALIZATIONS
     */
    NormalizeHpfDeclarations();
    NormalizeCodeForHpfc(s);

    /* here because the module was updated with some external declarations
     */
    init_host_and_node_entities(); 

    /*   ACTUAL COMPILATION
     */
    hpf_compiler(s, &host_stat, &node_stat);

    if (entity_main_module_p(module))
	add_pvm_init_and_end(&host_stat, &node_stat);

    declaration_with_overlaps_for_module(module);

    update_object_for_module(node_stat, node_module);
    update_object_for_module(entity_code(node_module), node_module);
    insure_declaration_coherency(node_module, node_stat);
    kill_statement_number_and_ordering(node_stat);
    
    update_object_for_module(host_stat, host_module);
    update_object_for_module(entity_code(host_module), host_module);
    insure_declaration_coherency(host_module, host_stat);
    kill_statement_number_and_ordering(host_stat);

    /*   PUT IN DB
     */
    put_generated_resources_for_module(s, host_stat, node_stat);

    /*   CLOSE
     */
    reset_resources_for_module();
}

/*---------------------------------------------------------------------
 *
 *  FUNCTIONS CALLED BY PIPSMAKE
 *
 */

/* void hpfc_init(name)
 * string name;
 *
 * what: initialize the hpfc status for a program.
 * input: the program (workspace) name.
 * output: none.
 * side effects:
 *  - the hpfc status is initialized and stored in the pips dbm.
 * bugs or features:
 *  - some callees are filtered out with a property, to deal with pipsmake.
 *  - ??? not all is in the hpfc status. common entities problem.
 */
void hpfc_init(name)
string name;
{
    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_init", "considering workspace %s\n", name);

    set_bool_property("PRETTYPRINT_HPFC", TRUE);
    set_bool_property("HPFC_FILTER_CALLEES", TRUE); /* drop hpfc specials */
    set_bool_property("GLOBAL_EFFECTS_TRANSLATION", FALSE);

    (void) make_empty_program(HPFC_PACKAGE);

    init_hpfc_status();
    save_hpfc_status();

    debug_off();
}

/* void hpfc_filter(name)
 * string name;
 *
 * what: filter the source code for module name. to be called by pipsmake.
 * how: call to a shell script, "hpfc_directives", that transforms 
 *      hpf directives in "special" subroutine calls to be parsed by 
 *      the fortran 77 parser.
 * input: the module name.
 * output: none.
 * side effects:
 *  - a new source code file is created for module name.
 *  - the old one is added a "-".
 * bugs or features:
 *  - ??? not all hpf syntaxes are managable this way.
 */
void hpfc_filter(name)
string name;
{
    string file_name = db_get_resource(DBR_SOURCE_FILE, name, TRUE);

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_filter", "considering module %s\n", name);

    system(concatenate("mv ", file_name, " ", file_name, "- ; ",
		       "$HPFC_TOOLS/hpfc_directives", 
		       " < ", file_name, "-", 
		       " > ", file_name, " ;",
		       NULL));

    DB_PUT_FILE_RESOURCE(DBR_HPFC_FILTERED, strdup(name), NO_FILE); /* fake */
    DB_PUT_FILE_RESOURCE(DBR_SOURCE_FILE, strdup(name), file_name);

    debug_off();
}

/* void hpfc_directives(name)
 * string name;
 *
 * what: deals with directives. to be called by pipsmake.
 * input: the name of the module.
 * output: none.
 * side effects: (many)
 *  - the module's code statement will be modified.
 *  - the hpf mappings and so are stored in the compiler status.
 * bugs or features:
 *  - fortran library, reduction and hpfc special functions are skipped.
 *  - ??? obscure problem with the update of common entities.
 */
void hpfc_directives(name)
string name;
{
    entity module = local_name_to_top_level_entity(name);
    statement s = (statement) db_get_resource(DBR_CODE, name, FALSE);

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_directives", "considering module %s\n", name);
    debug_on("HPFC_DIRECTIVES_DEBUG_LEVEL");

    if (!hpfc_entity_reduction_p(module) &&
	!hpf_directive_entity_p(module) &&
	!fortran_library_entity_p(module))
    {
	set_current_module_entity(module);
	load_hpfc_status();
	make_update_common_map(); 
	
	NormalizeCommonVariables(module, s);
	/* debug_print_referenced_entities(s); */
	build_full_ctrl_graph(s);
	handle_hpf_directives(s);

	clean_ctrl_graph();
	free_update_common_map(); 
	reset_current_module_entity();
	
	db_unput_a_resource(DBR_CODE, name);
	DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(name), s);

	save_hpfc_status();
    }

    DB_PUT_FILE_RESOURCE(DBR_HPFC_DIRECTIVES, strdup(name), NO_FILE);/* fake */

    debug_off(); debug_off();
}

/* void hpfc_compile(name)
 * string name;
 *
 * what: hpf compile module name. to be called by pipsmake.
 * input: the name of the module to compile.
 * output: none
 * side effects: (many)
 *  - creates the statements for the host and nodes.
 *  - store the generated resources.
 * bugs or features:
 *  - fortran library, reduction and hpfc special functions are skipped.
 *  - a fake file is put as the generated resource for such modules.
 */
void hpfc_compile(name)
string name;
{
    entity module = local_name_to_top_level_entity(name);

    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_compile", "considering module %s\n", name);

    if (!hpfc_entity_reduction_p(module) &&
	!hpf_directive_entity_p(module) &&
	!fortran_library_entity_p(module))
    {
	load_hpfc_status();
	set_current_module_entity(module);

	set_bool_property("PRETTYPRINT_COMMONS", FALSE); 

	compile_module(module);

	reset_current_module_entity();
	save_hpfc_status();
    }
    else
    {
	DB_PUT_FILE_RESOURCE(DBR_HPFC_HOST, strdup(name), NO_FILE); /* fake */
    }

    debug_off();
}

/* void hpfc_close(name)
 * string name;
 *
 * what: closes the hpf compiler execution. to be called by pipsmake.
 * input: the program (workspace) name.
 * output: none.
 * side effects:
 *  - deals with the commons.
 *  - generates global files.
 * bugs or features:
 *  - ??? COMMON should be managable thru pipsmake ad-hoc rules.
 */
void hpfc_close(name)
string name;
{
    debug_on("HPFC_DEBUG_LEVEL");
    debug(1, "hpfc_close", "considering %s\n", name);
 
    load_hpfc_status();
    
    set_bool_property("PRETTYPRINT_COMMONS", TRUE); /* commons compilation */
    gen_map(compile_common, get_the_commons());

    put_generated_resources_for_program(name);      /* global informations */

    close_hpfc_status();
    db_unput_resources(DBR_HPFC_STATUS);            /* destroy hpfc status */

    debug_off();
}

/*   that is all
 */
