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
#include "pipsdbm.h"      /* DB_PUT_FILE_RESOURCE is defined there */
#include "icfg.h"

#define ICFG_SCAN_INDENT 4

static bool CHECK = FALSE;
static bool PRINT_OUT = FALSE;
static FILE *fp;
static string caller_name;

void print_module_icfg(entity module)
{
    caller_name = "";
    module_to_icfg(0, module);
}

void module_to_icfg(margin, module)
int margin;
entity module;
{
    string module_name = module_local_name(module);
    statement s =(statement)db_get_memory_resource(DBR_CODE,module_name,TRUE);
    string filename = NULL;
    text r = make_text(NIL);

    if ( s == statement_undefined )
	pips_error("module_to_icfg","statement for module %s\n",module_name);
    else {
	if (margin==0) {
	    filename = strdup(concatenate
			      (db_get_current_program_directory(), 
			       "/", module_name,
			       get_bool_property(ICFG_IFs) ? ".icfgc" :
			       ( get_bool_property(ICFG_DOs) ? ".icfgl" : 
				".icfg") ,
			       NULL));

	    fp = safe_fopen(filename, "w");
	}

	switch (get_int_property (ICFG_DECOR)) {
	case ICFG_DECOR_NONE:
	    break;
	case ICFG_DECOR_TRANSFORMERS:
	    MERGE_TEXTS(r,get_text_transformers(module_name));
	    break;
	case ICFG_DECOR_PRECONDITIONS:
	{
	    /*
	       list ef = load_summary_effects(module);
	       transformer p = (transformer) db_get_memory_resource
	       (DBR_SUMMARY_PRECONDITION,
	       module_name,
	       FALSE);
	       
	       p = precondition_intra_to_inter (module, p, ef);
	       */
	    MERGE_TEXTS(r,get_text_preconditions(module_name));
	    break;
	}
	case ICFG_DECOR_PROPER_EFFECTS:
	    MERGE_TEXTS(r,get_text_proper_effects(module_name));
	    break;
	case ICFG_DECOR_CUMULATED_EFFECTS:
	    MERGE_TEXTS(r,get_text_cumulated_effects(module_name));
	    break;
	case ICFG_DECOR_REGIONS:
	    MERGE_TEXTS(r,get_text_regions(module_name));
	    break;
	case ICFG_DECOR_IN_REGIONS:
	    MERGE_TEXTS(r,get_text_in_regions(module_name));
	    break;
	case ICFG_DECOR_OUT_REGIONS:
	    MERGE_TEXTS(r,get_text_out_regions(module_name));
	    break;
	default:
	    pips_error("module_to_icfg",
		       "unknown ICFG decoration for module %s\n",
		       module_name);
	}
	
	print_text(fp, r);

	fprintf(fp,"%*s%s\n",margin,"",module_local_name(module));

	if (get_bool_property(ICFG_DEBUG)) 
	    fprintf(fp,"/modu");
  
	statement_to_icfg(margin+ICFG_SCAN_INDENT, s);

	if (margin == 0 ) {
	    safe_fclose(fp, filename);
	    DB_PUT_FILE_RESOURCE(strdup(DBR_ICFG_FILE),
				 strdup(module_name), filename);
	}
    }
}

void statement_to_icfg(margin, s)
int margin;
statement s;
{
    instruction_to_icfg(margin, statement_instruction(s));
}

void instruction_to_icfg(margin, instr)
int margin;
instruction instr;
{
    if (instruction_block_p(instr)) {
	block_to_icfg(margin, instruction_block(instr));
    }
    else if (instruction_test_p(instr)) {
	test_to_icfg(margin, instruction_test(instr));
    }
    else if (instruction_loop_p(instr)) {
	loop_to_icfg(margin, instruction_loop(instr));
    }
    else if (instruction_goto_p(instr)) { 
	goto_to_icfg(margin, instruction_goto(instr));	
    }
    else if (instruction_call_p(instr)) {
	call_to_icfg(margin, instruction_call(instr));
    }
    else if (instruction_unstructured_p(instr)) {
	unstructured_to_icfg(margin, instruction_unstructured(instr));
    }
    else 
	pips_error("instruction_to_icfg", "unexpected tag\n");
}

void block_to_icfg(margin, block)
int margin;
list block;
{
    MAPL(pm, {   
	statement s = STATEMENT(CAR(pm));
	statement_to_icfg(margin, s);
    }, block);
}

void test_to_icfg(margin, test_instr)
int margin;
test test_instr;
{
    expression cond   = test_condition(test_instr);
    statement s_true  = test_true(test_instr);
    statement s_false = test_false(test_instr);

    if ( get_bool_property(ICFG_IFs)  && !CHECK) {

	CHECK = TRUE;
	if ( !PRINT_OUT )
	    expression_to_icfg(margin, cond);
	if ( !PRINT_OUT )
	    statement_to_icfg(margin, s_true);
	if ( !PRINT_OUT )
	    statement_to_icfg(margin, s_false);
	CHECK = FALSE;

	if ( PRINT_OUT ) {
	    PRINT_OUT = FALSE;

	    fprintf(fp,"%*sIF\n",margin,"");
	    expression_to_icfg(margin+ICFG_SCAN_INDENT, cond);
	    fprintf(fp,"%*sTHEN\n",margin,"");
	    statement_to_icfg(margin+ICFG_SCAN_INDENT, s_true);
	    fprintf(fp,"%*sELSE\n",margin,"");
	    statement_to_icfg(margin+ICFG_SCAN_INDENT, s_false);
	    fprintf(fp,"%*sENDIF\n",margin,"");
	}
    }
    else {
	expression_to_icfg(margin, cond);
	statement_to_icfg(margin, s_true);
	statement_to_icfg(margin, s_false);
    }
}

void loop_to_icfg(margin, loop_instr)
int margin;
loop loop_instr;
{
    range r = loop_range(loop_instr);
    statement s = loop_body(loop_instr);

    if ( get_bool_property(ICFG_DOs) && !CHECK) {

	CHECK = TRUE;
	if ( !PRINT_OUT )
	    range_to_icfg(margin, r);
	if ( !PRINT_OUT )
	    statement_to_icfg(margin, s);
	CHECK = FALSE;

	if ( PRINT_OUT ) {
	    PRINT_OUT = FALSE;

	    fprintf(fp,"%*sDO",margin,"");
	    range_to_icfg(margin+ICFG_SCAN_INDENT, r);
	    fprintf(fp,"\n");
	    statement_to_icfg(margin+ICFG_SCAN_INDENT, s);
	    fprintf(fp,"%*sENDDO\n",margin,"");
	}
    }
    else {
	range_to_icfg(margin, r);
	statement_to_icfg(margin, s);
    }

}

void goto_to_icfg(margin, goto_instr)
int margin;
statement goto_instr;
{
    statement_to_icfg(margin, goto_instr);
}

void call_to_icfg(margin, call_instr)
int margin;
call call_instr;
{
    entity module = call_function(call_instr);
    string module_name = module_local_name(module);

    if (get_bool_property(ICFG_DEBUG))
	fprintf(fp,"/call %s", module_name);

    MAPL(pa, {
	 expression_to_icfg(margin, EXPRESSION(CAR(pa))); },
	 call_arguments(call_instr) );

    if (value_code_p(entity_initial(module))) {
	if ( CHECK ) {
	    PRINT_OUT = TRUE;
	    return;
	} else {
	    
	    /*
	       list ef = load_summary_effects(module);
	       transformer p = (transformer) db_get_memory_resource
	       (DBR_PRECONDITIONS,
	       module_name,
	       FALSE);
	       p = precondition_intra_to_inter (module, p, ef);
	       */
	    
	    module_to_icfg(margin, module);
	}
    }
}

void unstructured_to_icfg(margin, u)
int margin;
unstructured u;
{
    control c;

    debug(8,"unstructured_to_icfg","begin\n");

    pips_assert("unstructured_to_icfg", u!=unstructured_undefined);

    c = unstructured_control(u);
    if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
	/* there is only one statement in u; no need for a fix-point */
	debug(8,"unstructured_to_icfg","unique node\n");
	statement_to_icfg(margin, control_statement(c));
    }
    else {
	/* Do not try anything clever! God knows what may happen in
	   unstructured code. Postcondition post is not computed recursively
	   from its components but directly derived from u's transformer.
	   Preconditions associated to its components are then computed
	   independently, hence the name unstructured_to_icfgS
	   instead of unstructured_to_icfg */
	/* propagate as precondition an invariant for the whole 
	   unstructured u */

	debug(8,"unstructured_to_icfg",
	      "complex: based on transformer\n");
	unstructured_to_icfgs(margin, u) ;

    }

    debug(8,"unstructured_to_icfg","end\n");
}

void unstructured_to_icfgs(margin, u)
int margin;
unstructured u ;
{
    cons *blocs = NIL ;
    control ct = unstructured_control(u) ;

    debug(8,"unstructured_to_icfgs","begin\n");

    /* SHARING! Every statement gets a pointer to the same precondition!
       I do not know if it's good or not but beware the bugs!!! */
    CONTROL_MAP(c, {
	statement st = control_statement(c) ;
	statement_to_icfg(margin, st) ;
    }, ct, blocs) ;

    gen_free_list(blocs) ;

    debug(8,"unstructured_to_icfgs","end\n");
}

void expression_to_icfg(margin, expr)
int margin;
expression expr;
{
    if (get_bool_property(ICFG_DEBUG)) 
	fprintf(fp,"/expr");

    syntax_to_icfg(margin, expression_syntax(expr));
}

void syntax_to_icfg(margin, synt)
int margin;
syntax synt;
{
    if (syntax_reference_p(synt)) {
	reference_to_icfg(margin, syntax_reference(synt));
    }
    else if (syntax_range_p(synt)) {
	range_to_icfg(margin, syntax_range(synt));
    }
    else if (syntax_call_p(synt)) {
	call_to_icfg(margin, syntax_call(synt));
    }
    else {
	pips_error("expression_to_icfg", "tag inconnu");
    }
}
    
void range_to_icfg(margin, rng)
int margin;
range rng;
{
    expression_to_icfg(margin, range_lower(rng));
    expression_to_icfg(margin, range_upper(rng));
    expression_to_icfg(margin, range_increment(rng));
}

void reference_to_icfg(margin, ref)
int margin;
reference ref;
{
    MAPL(pi, { expression_to_icfg(margin, EXPRESSION(CAR(pi))); },
	 reference_indices(ref))
}
