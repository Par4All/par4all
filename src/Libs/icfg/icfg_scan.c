/* 
   icfg_scan.c
   module_to_icfg(0, mod) recursively to_icfgs module "mod" and its callees
   and writes its icfg in indented form
*/
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "list.h"
#include "ri.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "properties.h"  /* get_bool_property */
#include "misc.h"
#include "database.h"
#include "effects.h"
#include "regions.h"
#include "resources.h"
#include "semantics.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "pipsdbm.h"      /* DB_PUT_FILE_RESOURCE is defined there */
#include "icfg.h"

#define ICFG_SCAN_INDENT 4

static bool CHECK = FALSE;
static bool PRINT_OUT = FALSE;
static int current_margin;
static FILE *fp;

/* We want to keep track of the current statement inside the recurse */
DEFINE_LOCAL_STACK(current_stmt, statement)

#define MAX_LINE_LENGTH 256

static void print_icfg_file(string module_name)
{
    string filename = NULL;
    string localfilename = NULL;
    FILE *f_called;
    char buf[MAX_LINE_LENGTH];

    /* create filename */
    localfilename = strdup(concatenate
			   (module_name,
			    get_bool_property(ICFG_IFs) ? ".icfgc" :
			    ( get_bool_property(ICFG_DOs) ? ".icfgl" : 
			     ".icfg") ,
			    NULL));
    filename = strdup(concatenate
		      (db_get_current_workspace_directory(), 
		       "/", localfilename, NULL));

    pips_debug (2, "Inserting ICFG for module %s\n", module_name);

    /* Get the Icfg from the callee */
    f_called = safe_fopen (filename, "r");
    while (fgets (buf, MAX_LINE_LENGTH, f_called))
	fprintf(fp,"%*s%s", current_margin ,"",buf);
    safe_fclose (f_called, filename);
}

static bool call_filter(call c)
{
    entity e_callee = call_function(c);
    string callee_name = module_local_name(e_callee);

    /* current_stmt_head() */
    pips_debug (5,"called entity is %s\n", entity_name(e_callee));

    /* If this is a "real function" (defined in the code elsewhere) */
    if (value_code_p(entity_initial(e_callee))) {
	text r = make_text(NIL);

	entity e_caller = get_current_module_entity();
	reset_current_module_entity();

	switch (get_int_property (ICFG_DECOR)) {
	case ICFG_DECOR_NONE:
	    break;
	case ICFG_DECOR_COMPLEXITIES:
	    MERGE_TEXTS(r,get_text_complexities(callee_name));
	    break;
	case ICFG_DECOR_TRANSFORMERS:
	    MERGE_TEXTS(r,get_text_transformers(callee_name));
	    break;
	case ICFG_DECOR_PRECONDITIONS:
	{
	    /* summary effects for the callee */
	    list seffects_callee = load_summary_effects(e_callee);
	    /* caller preconditions */
	    transformer caller_prec;
	    /* callee preconditions */
	    transformer call_site_prec;


	    
	    set_cumulated_effects_map
		(effectsmap_to_listmap((statement_mapping)
				       db_get_memory_resource
				       (DBR_CUMULATED_EFFECTS,
					module_local_name(e_caller), TRUE)));
	    set_semantic_map((statement_mapping)
			     db_get_memory_resource
			     (DBR_PRECONDITIONS,
			      module_local_name(e_caller),
			      TRUE) );

	    /* load caller preconditions */
	    caller_prec = load_statement_semantic(current_stmt_head());

	    set_current_module_statement (current_stmt_head());

	    /* first, we deal with the caller */
	    set_current_module_entity(e_caller);
	    /* create htable for old_values ... */
	    module_to_value_mappings(e_caller);
	    /* add to preconditions the links to the callee formal params */
	    caller_prec = add_formal_to_actual_bindings (c, caller_prec);
	    /* transform the preconditions */
	    call_site_prec = precondition_intra_to_inter (e_callee,
						caller_prec,
						seffects_callee);
	    /* translate_global_values(e_caller, call_site_prec); */
	    reset_current_module_entity();

	    /* Now deal with the callee */
	    set_current_module_entity(e_callee);
	    /* Set the htable with its varaibles because now we work
	       in tis referential*/
	    module_to_value_mappings(e_callee); 
	    reset_current_module_entity();

	    /* Then print the text for the caller preconditions */
	    set_current_module_entity(e_caller);
	    MERGE_TEXTS(r,text_transformer(call_site_prec));
	    reset_current_module_entity();

	    reset_current_module_statement();
	    reset_cumulated_effects_map();
	    reset_semantic_map();


	    break;
	}
	case ICFG_DECOR_PROPER_EFFECTS:
	    MERGE_TEXTS(r,get_text_proper_effects(callee_name));
	    break;
	case ICFG_DECOR_CUMULATED_EFFECTS:
	    MERGE_TEXTS(r,get_text_cumulated_effects(callee_name));
	    break;
	case ICFG_DECOR_REGIONS:
	    MERGE_TEXTS(r,get_text_regions(callee_name));
	    break;
	case ICFG_DECOR_IN_REGIONS:
	    MERGE_TEXTS(r,get_text_in_regions(callee_name));
	    break;
	case ICFG_DECOR_OUT_REGIONS:
	    MERGE_TEXTS(r,get_text_out_regions(callee_name));
	    break;
	default:
	    pips_error("module_to_icfg",
		       "unknown ICFG decoration for module %s\n",
		       callee_name);
	}

	set_current_module_entity(e_caller);
	print_text(fp, r);
	print_icfg_file (callee_name);
	free_text(r);
    }
    return TRUE;
}

static bool loop_filter (loop l)
{
    pips_debug (5,"Loop begin\n");

    if (get_bool_property(ICFG_DOs)) {
	fprintf(fp,"%*sDO\n", current_margin, "");
	current_margin += ICFG_SCAN_INDENT;
    }
    return TRUE;
}

static bool loop_rewrite (loop l)
{
    pips_debug (5,"Loop end\n");

    if (get_bool_property(ICFG_DOs)) {
	current_margin -= ICFG_SCAN_INDENT;
	fprintf(fp,"%*sENDDO\n", current_margin, "");
    }
    return TRUE;
}

static bool test_filter (test l)
{
    pips_debug (5, "Test begin\n");

    if (get_bool_property(ICFG_IFs)) {
	fprintf(fp,"%*sIF\n", current_margin, "");
	current_margin += ICFG_SCAN_INDENT;
    }
    return TRUE;
}

static bool test_rewrite (test l)
{
    pips_debug (5, "Test end\n");

    if (get_bool_property(ICFG_IFs)) {
	current_margin -= ICFG_SCAN_INDENT;
	fprintf(fp,"%*sENDIF\n", current_margin, "");
    }
    return TRUE;
}

void print_module_icfg(entity module)
{
    string filename = NULL;
    string localfilename = NULL;
    string module_name = module_local_name(module);
    statement s =(statement)db_get_memory_resource(DBR_CODE,module_name,TRUE);

    set_current_module_entity (module);

    /* Build filename */
    localfilename = strdup(concatenate
			   (module_name,
			    get_bool_property(ICFG_IFs) ? ".icfgc" :
			    (get_bool_property(ICFG_DOs) ? ".icfgl" : 
			     ".icfg") ,
			    NULL));
    filename = strdup(concatenate
		      (db_get_current_workspace_directory(), 
		       "/", localfilename, NULL));

    fp = safe_fopen(filename, "w");
    make_current_stmt_stack();

    fprintf(fp,"%s\n",module_name);

    current_margin = ICFG_SCAN_INDENT;

    gen_multi_recurse
	(s,
	 statement_domain, current_stmt_filter, current_stmt_rewrite,
	 call_domain,call_filter,gen_null,
	 loop_domain,loop_filter, loop_rewrite,
	 test_domain,test_filter, test_rewrite,
	 NULL);

    pips_assert("empty stack", current_stmt_empty_p());
    
    safe_fclose(fp, filename);
    free(filename);
    DB_PUT_FILE_RESOURCE(strdup(DBR_ICFG_FILE),
			 strdup(module_name), localfilename);
    free_current_stmt_stack();
    reset_current_module_entity();
}
