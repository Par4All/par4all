/*
 * HPFC
 * 
 * Fabien Coelho, May 1993
 *
 * SCCS Stuff:
 * $RCSfile: compile.c,v $ ($Date: 1994/06/03 14:14:26 $) version $Revision$, got on %D%, %T%
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

#include "types.h"
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
#include "pipsdbm.h"
#include "resources.h"
#include "effects.h"
#include "semantics.h"
#include "regions.h"
#include "hpfc.h"
#include "defines-local.h"

/* external functions */
extern char    *getenv();
extern void AddEntityToDeclarations(entity e, entity f); /* in syntax.h */

/*
 * hpfcompile
 *
 * Compiler call
 */
void hpfcompile(module_name)
char *module_name;
{
    entity          
	module = local_name_to_top_level_entity(module_name);
    statement   
	module_stat,
	hoststat,
	nodestat;
    FILE 
	*hostfile,
	*nodefile,
	*parmfile,
	*initfile,
	*normfile;
    string
	hostfilename,
	nodefilename,
	parmfilename,
	initfilename,
	normfilename;

    debug_on("HPFC_DEBUG_LEVEL");

    debug(3,"hpfcompile","module: %s\n",module_name);

    set_current_module_entity(module);
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, FALSE));
    module_stat = get_current_module_statement();

/*    set_cumulated_effects_map
 *	((statement_mapping) 
 *	 db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, FALSE));
 *    set_proper_effects_map
 *	((statement_mapping) 
 *	 db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, FALSE));
 */
    /*
     * Preconditions are loaded
     */
    set_precondition_map
	((statement_mapping)
	 db_get_memory_resource(DBR_PRECONDITIONS, module_name, FALSE));
    /*
     * Postconditions are computed
     */
    set_postcondition_map
	(compute_postcondition(module_stat, 
			       MAKE_STATEMENT_MAPPING(),
			       get_precondition_map()));
    /*
     * Regions are loaded
     */
    set_local_regions_map
	(effectsmap_to_listmap((statement_mapping)
	 db_get_memory_resource(DBR_REGIONS, module_name, FALSE)));
    
    /*
     * Initialize mappings
     */
    make_hpfc_current_mappings();
    only_io_mapping_initialize(module_stat);
    
    /* HPFC-PACKAGE is used to put dummy variables in
     */
    (void) make_empty_program(HPFC_PACKAGE);

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

    debug(1, "hpfccompile", 
	  "building %s.hpf in %s\n",
	  db_get_current_module_name(),
	  db_get_current_program_directory());

    system(concatenate("$UTILDIR/hpfc_filter < ",
		       db_get_file_resource(DBR_SOURCE_FILE, module_name, TRUE),
		       " > ",
		       db_get_current_program_directory(),
		       "/",
		       db_get_current_module_name(),
		       ".hpf",
		       NIL));

    InitializeGlobalVariablesOfHpfc();
    hpfc_init_run_time_entities();
    init_overlap_management();
    ReadHpfDir(module_name);
    NormalizeHpfDeclarations();
    NormalizeCodeForHpfc(module_stat);
   
    normfilename = 
	strdup(concatenate(db_get_current_program_directory(), 
			   "/",
			   module_local_name(get_current_module_entity()),
			   ".norm", 
			   NULL));

    normfile = (FILE *) safe_fopen(normfilename, "w");
    hpfc_print_code(normfile, get_current_module_entity(), module_stat);
    safe_fclose(normfile, normfilename);

    init_host_and_node_entities();
    /* init_pvm_based_intrinsics(); // old */
    
    hoststat = statement_undefined;
    nodestat = statement_undefined;

    hpfcompiler(module_stat, &hoststat, &nodestat);
    add_pvm_init_and_end(&hoststat, &nodestat);

    /*
    DeduceGotos(hoststat, hostgotos);
    DeduceGotos(nodestat, nodegotos);
    */
    declaration_with_overlaps();
    close_overlap_management();

    /*
     * output
     */
    
    hostfilename = 
	strdup(concatenate(db_get_current_program_directory(),
			   "/host.f", NULL));
    nodefilename = 
	strdup(concatenate(db_get_current_program_directory(),
			   "/node.f", NULL));
    parmfilename = 
	strdup(concatenate(db_get_current_program_directory(),
			   "/parameters.h", NULL));
    initfilename = 
	strdup(concatenate(db_get_current_program_directory(),
			   "/init_param.f", NULL));


    hostfile = (FILE *) safe_fopen(hostfilename, "w");
    hpfc_print_code(hostfile, host_module, hoststat);
    safe_fclose(hostfile, hostfilename);
    system(concatenate("$UTILDIR/hpfc_add_includes ", 
		       hostfilename, 
		       " ", 
		       db_get_file_resource(DBR_SOURCE_FILE, module_name, TRUE),
		       NIL));

    nodefile = (FILE *) safe_fopen(nodefilename, "w");
    hpfc_print_code(nodefile, node_module, nodestat);
    safe_fclose(nodefile, nodefilename);
    system(concatenate("$UTILDIR/hpfc_add_includes ", 
		       nodefilename,
		       " ", 
		       db_get_file_resource(DBR_SOURCE_FILE, module_name, TRUE),
		       NIL));

    parmfile = (FILE *) safe_fopen(parmfilename, "w");
    create_parameters_h(parmfile);
    safe_fclose(parmfile, parmfilename);

    initfile = (FILE *) safe_fopen(initfilename, "w");
    create_init_common_param(initfile);
    safe_fclose(initfile, initfilename);

    ifdebug(1)
    {
	fprintf(stderr, "Result of HPFC:\n");
	fprintf(stderr, "-----------------\n");
	hpfc_print_code(stderr, host_module, hoststat);
	fprintf(stderr, "-----------------\n");
	hpfc_print_code(stderr, node_module, nodestat);
	fprintf(stderr, "-----------------\n");
	create_parameters_h(stderr);
	fprintf(stderr, "-----------------\n");
	create_init_common_param(stderr);
	fprintf(stderr, "-----------------\n");
    }

/*    DB_PUT_FILE_RESOURCE(DBR_xxx, strdup(module_name), filename);
 */
    
    debug(4,"hpfcompile","end of procedure\n");

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_effects_map();
    reset_proper_effects_map();
    reset_local_regions_map();

    free_postcondition_map();
    reset_postcondition_map();

    free_hpfc_current_mappings();
    debug_off();
}

/*
 * ReadHpfDir
 */
void ReadHpfDir(module_name)
string module_name;
{
    debug(8,"ReadHpfDir","module: %s\n",module_name);
    
    /* filter */
    hpfcparser(module_name);
    
}


/*
 * InitializeGlobalVariablesOfHpfc
 *
 * Global variable initialization
 */	 
void InitializeGlobalVariablesOfHpfc()
{
    debug(8, "InitializeGlobalVariablesOfHpfc", "Hello !\n");

    hpfc_init_unique_numbers();
    reset_hpf_object_lists();
    make_new_declaration_map();
    make_align_map();
    make_distribute_map();
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
void init_host_and_node_entities()
{
    host_module = make_empty_program(HOST_NAME);
    node_module = make_empty_program(NODE_NAME);


    hostgotos = MAKE_STATEMENT_MAPPING();
    nodegotos = MAKE_STATEMENT_MAPPING();

    /*
     * ??? be carefull, sharing of structures...
     * between the compiled module, the host and the node.
     */

    make_host_node_maps();

    MAPL(ce,
     {
	 entity 
	     e = ENTITY(CAR(ce));
	 type
	     t = entity_type(e);

	 /* 
	  * parameters are selected. I think they may be either
	  * functional of variable (if declared...) FC 15/09/93
	  */

	 if ((type_variable_p(t)) ||
	     ((storage_rom_p(entity_storage(e))) &&
	      (value_symbolic_p(entity_initial(e)))))
	     AddEntityToHostAndNodeModules(e);
	 else
	 if (type_functional_p(t))
	 {
	     AddEntityToDeclarations(e, host_module);
	     AddEntityToDeclarations(e, node_module);
	 }
	 
     },
	 entity_declarations(get_current_module_entity())); 
    
    NewDeclarationsOfDistributedArrays();    

    ifdebug(3)
    {
	debug_off();
	fprintf(stderr,"[init_host_and_node_entities]\n old declarations:\n");
	print_text(stderr,text_declaration(get_current_module_entity()));
	fprintf(stderr, "new declarations,\nhost_module:\n");
	print_text(stderr,text_declaration(host_module));
	fprintf(stderr,"node_module:\n");
	print_text(stderr,text_declaration(node_module));
	debug_on("HPFC_DEBUG_LEVEL");
    }
}
