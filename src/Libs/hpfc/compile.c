/*
 * HPFC
 * 
 * Fabien Coelho, May 1993
 *
 */

/*
 * included files, from C libraries, newgen and pips libraries.
 */

#include <stdio.h>
#include <string.h>

extern int fprintf();
extern int vfprintf();
extern int system();

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "hpf.h"
#include "hpf_private.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
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
/*    CurrentFunction = module;  is that not necessary any more? */
    module_stat = (statement)
	db_get_memory_resource(DBR_CODE, module_name, FALSE);


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

    system(concatenate("$UTILDIR/filter-hpf < ",
		       db_get_file_resource(DBR_SOURCE_FILE, module_name, TRUE),
		       " > ",
		       db_get_current_program_directory(),
		       "/",
		       db_get_current_module_name(),
		       ".hpf",
		       NIL));

    InitializeGlobalVariablesOfHpfc();
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
    init_pvm_based_intrinsics();
    
    hoststat = statement_undefined;
    nodestat = statement_undefined;

    hpfcompiler(module_stat, &hoststat, &nodestat);
    add_pvm_init_and_end(&hoststat, &nodestat);

    DeduceGotos(hoststat,hostgotos);
    DeduceGotos(nodestat,nodegotos);
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
    hpfc_print_code(hostfile, hostmodule, hoststat);
    safe_fclose(hostfile, hostfilename);
    system(concatenate("$UTILDIR/add-includes ", hostfilename, NIL));

    nodefile = (FILE *) safe_fopen(nodefilename, "w");
    hpfc_print_code(nodefile, nodemodule, nodestat);
    safe_fclose(nodefile, nodefilename);
    system(concatenate("$UTILDIR/add-includes ", nodefilename, NIL));

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
	hpfc_print_code(stderr, hostmodule, hoststat);
	fprintf(stderr, "-----------------\n");
	hpfc_print_code(stderr, nodemodule, nodestat);
	fprintf(stderr, "-----------------\n");
	create_parameters_h(stderr);
	fprintf(stderr, "-----------------\n");
	create_init_common_param(stderr);
	fprintf(stderr, "-----------------\n");
    }

/*    DB_PUT_FILE_RESOURCE(DBR_xxx, strdup(module_name), filename);*/
    
    debug(4,"hpfcompile","end of procedure\n");
    reset_current_module_entity();
    reset_hpfc_static_mappings();
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
    debug(8,"InitializeGlobalVariablesOfHpfc","Hello !\n");

    uniqueintegernumber=0;
    uniquefloatnumber=0;
    uniquelogicalnumber=0;
    uniquecomplexnumber=0;

    hpfnumber=MAKE_ENTITY_MAPPING();
    hpfalign=MAKE_ENTITY_MAPPING();
    hpfdistribute=MAKE_ENTITY_MAPPING();

    distributedarrays=NULL;
    templates=NULL;
    processors=NULL;

    /* and others? */
}


/*
 * init_host_and_node_entities
 *
 * both host and node modules are initialized with the same
 * declarations than the compiled module, but the distributed arrays
 * declarations... which are not declared in the case of the hostmodule,
 * and the declarations of which are modified in the nodemodule
 * (call to NewDeclarationsOfDistributedArrays)...
 */
void init_host_and_node_entities()
{
    hostmodule = make_empty_program(HOST_NAME);
    nodemodule = make_empty_program(NODE_NAME);


    hostgotos = MAKE_STATEMENT_MAPPING();
    nodegotos = MAKE_STATEMENT_MAPPING();

    /*
     * ??? be carefull, sharing of structures...
     * between the compiled module, the host and the node.
     */

    oldtonewhostvar = MAKE_ENTITY_MAPPING();
    oldtonewnodevar = MAKE_ENTITY_MAPPING();
    newtooldhostvar = MAKE_ENTITY_MAPPING();
    newtooldnodevar = MAKE_ENTITY_MAPPING();

    newdeclarations = MAKE_ENTITY_MAPPING();

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
     },
	 entity_declarations(get_current_module_entity())); 
    
    NewDeclarationsOfDistributedArrays();    


    /* overloaded basic type, why not? */
    e_MYPOS = make_scalar_entity("MYPOS", 
				 module_local_name(nodemodule),
				 MakeBasic(is_basic_overloaded)); 

    variable_dimensions(type_variable(entity_type(e_MYPOS))) =
	CONS(DIMENSION,
	     make_dimension(int_to_expression(1),
			    int_to_expression(7)),
	CONS(DIMENSION,
	     make_dimension(int_to_expression(1),
			    int_to_expression(1024)), /* why not ? */
	     NIL));

    ifdebug(3)
    {
	debug_off();
	fprintf(stderr,"[init_host_and_node_entities]\n old declarations:\n");
	print_text(stderr,text_declaration(get_current_module_entity()));
	fprintf(stderr, "new declarations,\nhostmodule:\n");
	print_text(stderr,text_declaration(hostmodule));
	fprintf(stderr,"nodemodule:\n");
	print_text(stderr,text_declaration(nodemodule));
	debug_on("HPFC_DEBUG_LEVEL");
    }
}
