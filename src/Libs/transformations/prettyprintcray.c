#include <stdio.h>
#include <string.h>
#include <values.h>

#include "genC.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "pipsdbm.h"
#include "prettyprint.h"

#include "top-level.h"
#include "resources.h"
#include "control.h" /* for macro CONTROL_MAP() */

#include "transformations.h"

bool same_entity_name_p(entity e1, entity e2)
{
    return(strcmp(entity_name(e1), entity_name(e2))==0);
}

bool entity_in_list(entity ent, cons *ent_l)
{
    MAP(ENTITY, e, {
	if (same_entity_p(ent, e)) {
	    debug(9, "entity_in_list", "entity %s found\n",
		  entity_local_name(ent));
	    return TRUE;
	}
    }, ent_l);

    return FALSE;
}

/* returns l1 after elements of l2 but not of l1 have been appended to l1. */
/* l2 is freed */
list concat_new_entities(list l1, list l2)
{
    list new_l2=NIL;

    MAPL(le, {
	entity e = ENTITY(CAR(le));

	if (!entity_in_list(e, l1)) {
	    new_l2 = gen_nconc(new_l2, CONS(ENTITY, e, NIL));
	}
    }, l2);
    gen_free_list(l2);
    return(gen_nconc(l1, new_l2));
}

/* returns a list of all entities which:
 * - are concerned with cumulated effects (cfx) of the loop_body
 * - and are member of loop_locals(lp)
 * makes sure the loop index is in the list
 */
list real_loop_locals(loop lp, effects cfx)
{
    list rll= NIL;

    MAPL(ce, 
     { 
	effect 
	    eff = EFFECT(CAR(ce));
	entity 
	    ent = reference_variable(effect_reference(eff));

	if (!entity_in_list(ent, rll)
	    && entity_in_list(ent,loop_locals(lp)) ) 
	{
	    /* ent is a real loop local */
	    debug(7, "real_loop_locals", "real loop local: %s\n",
		  entity_local_name(ent));
	    rll = CONS(ENTITY, ent, rll);
	}
    }, effects_effects(cfx));

    if( !entity_in_list(loop_index(lp), rll) ) 
    {
	rll= CONS(ENTITY, loop_index(lp), rll);
    }
    return(rll);
}

/*
 * recursively concatenates all real loop locals of all enclosed loops.
 * filters redondant entities
 */
list all_enclosed_scope_variables(statement stmt)
{
    instruction instr = statement_instruction(stmt);
    list ent_l= NIL;

    switch(instruction_tag(instr)) {
      case is_instruction_block :
	MAPL(stmt_l, {
	    statement st=STATEMENT(CAR(stmt_l));

	    ent_l= concat_new_entities(ent_l, all_enclosed_scope_variables(st));
	}, instruction_block(instr));
	break;
      case is_instruction_loop : {
	  loop lp= instruction_loop(instr);
	  statement lpb= loop_body(lp);
	  effects cfx = stmt_to_fx(lpb, get_rw_effects() );

	  pips_assert("all_enclosed_scope_variables", cfx != effects_undefined);
	  ent_l= concat_new_entities(real_loop_locals(lp, cfx),
				     all_enclosed_scope_variables(lpb));
	  break;
      }
      case is_instruction_test : {
	  test tst= instruction_test(instr);
	  list l1, l2;

	  l1= all_enclosed_scope_variables(test_true(tst));
	  l2= all_enclosed_scope_variables(test_false(tst));
	  ent_l= concat_new_entities(l1, l2);

	  break;
      }
      case is_instruction_unstructured : {
	  list blocs = NIL;

	  CONTROL_MAP(ctl, {
	      statement st = control_statement(ctl);

	      ent_l= concat_new_entities(all_enclosed_scope_variables(st), 
					 ent_l);	
	  }, unstructured_control(instruction_unstructured(instr)), blocs);

	  gen_free_list(blocs);
	  break;
      }
      case is_instruction_call :
	break;
      case is_instruction_goto :
	default : 
	pips_error("all_enclosed_scope_variables", 
		   "Bad instruction tag");
    }
    return(ent_l);
}

/* lp_stt must be a loop statement */
text text_microtasked_loop(entity module, int margin, statement lp_stt)
{
    text txt;
    unformatted u;
    effects fx;
    list wordl;
    int np = 0;
    loop lp = instruction_loop(statement_instruction(lp_stt));
    cons *lp_shared = NIL;
    list ent_l= all_enclosed_scope_variables(lp_stt);

    /* initializations */
    txt = make_text(NIL);

    wordl = CHAIN_SWORD(NIL, "DO ALL ");

    fx = stmt_to_fx(loop_body(lp), get_rw_effects());

    /* generate arguments for PRIVATE */
    /* nb: ent_l should contain entities only ones. */
    wordl = CHAIN_SWORD(wordl, "PRIVATE(");
    np=0;
    MAPL( el, { 
	entity ent = ENTITY(CAR(el));

	/* What about arrays? nothing special?? */
	/* if (!ENDP(reference_indices(effect_reference(eff)))) */

	if (np>0)
	    wordl = CHAIN_SWORD(wordl, ",");
	wordl = CHAIN_SWORD(wordl, entity_local_name(ent)) ;
	np++;
    }, ent_l);

    wordl = CHAIN_SWORD(wordl, ") ");

    ifdebug(6)
    {
	fprintf(stderr, 
		"[text_microtasked_loop] loop locals of %s: ",
		entity_minimal_name(loop_index(lp)));

	MAPL(ce,
	 {
	     fprintf(stderr, "%s ", entity_minimal_name(ENTITY(CAR(ce))));
	 },
	     loop_locals(lp));

	fprintf(stderr, "\nent_l content is: ");
	
	MAPL(ce,
	 {
	     fprintf(stderr, "%s ", entity_minimal_name(ENTITY(CAR(ce))));
	 },
	     ent_l);

	fprintf(stderr, "\n");
    }

    /* generate arguments for SHARED */
    wordl = CHAIN_SWORD(wordl, "SHARED(");
    np=0;
    MAPL(ce, 
     { 
	effect eff = EFFECT(CAR(ce));
	entity ent = reference_variable(effect_reference(eff));

/*	if(ENDP(ce)) user_log("ce is NIL !!\n");
	if(ENDP(CDR(ce))) user_log("CDR(ce) is NIL\n");
	else user_log("CDR(ce) is *not* NIL\n");
 */
	/* What about arrays? nothing special?? */
	/* if (!ENDP(reference_indices(effect_reference(eff)))) */

	/*
	 * ent_l added, in order to use the newly computed << good >> effects, 
	 * and to warrant the fact that no variables should be declared
	 * at the same time private and shared.
	 * same_entity_p added also, just in case.
	 *
	 * FC 28/09/93
	 */
	if ((!entity_in_list(ent, loop_locals(lp))) &&
	    (!entity_in_list(ent, ent_l)) &&
	    (!same_entity_p(ent, loop_index(lp))) &&
	    (!entity_in_list(ent, lp_shared)))
	{
	    /* ent is a new shared entity */
	    if (np>0)
		wordl = CHAIN_SWORD(wordl, ",");
	    lp_shared = CONS(ENTITY, ent, lp_shared);
	    wordl = CHAIN_SWORD(wordl, entity_local_name(ent)) ;
	    np++;
	}
	/* What to do when no shared? */
	    
    }, effects_effects(fx));

    wordl = CHAIN_SWORD(wordl, ") ");

    gen_free_list(lp_shared);
    gen_free_list(ent_l);

    u = make_unformatted("CMIC$", 0, 0, wordl) ;

    /* format u */


    ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_unformatted, u));
    return(txt);
}

/* lp_stt must be a loop statement */
text text_vectorized_loop(entity module, int margin, statement lp_stt)
{
    text txt;
    unformatted u;
    list wordl;

    txt = make_text(NIL);

    wordl = CHAIN_SWORD(NIL, "IVDEP ");

    u = make_unformatted("CDIR$", 0, 0, wordl) ;

    ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_unformatted, u));
    return(txt);
}


text text_cray(entity module, int margin, statement stat)
{
    text txt = text_undefined;

    if (instruction_loop_p(statement_instruction(stat))) {
	loop lp = instruction_loop(statement_instruction(stat));
	statement body = loop_body(lp);

	switch(execution_tag(loop_execution(lp))) {
	  case (is_execution_sequential):
	    txt = make_text(NIL);
	    break;
	  case (is_execution_parallel):
	    if(instruction_assign_p(statement_instruction(body))) {
		/* vector loop */
		txt = text_vectorized_loop(module, margin, stat);
	    }
	    else {
		/* assumes that all loops are CMIC$ !! */
		txt = text_microtasked_loop(module, margin, stat);
	    }
	    break;
	}
    }
    else {
	txt = make_text(NIL);
    }

    return(txt);
}


bool print_parallelizedcray_code(char *mod_name)
{
    entity module = local_name_to_top_level_entity(mod_name);
    statement mod_stat = statement_undefined;

    /* push prettyprint style */
    string pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
    set_string_property(PRETTYPRINT_PARALLEL, "cray");

    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, TRUE);
    
    /* We need to recompute proper effects and cumulated effects */
    init_proper_rw_effects();
    rproper_effects_of_statement(mod_stat);

    init_rw_effects();
    rcumulated_effects_of_statement(mod_stat);

    set_current_module_entity(module);

    debug_on("PRETTYPRINT_DEBUG_LEVEL");

    init_prettyprint(text_cray);

    make_text_resource(mod_name, 
		       DBR_PARALLELPRINTED_FILE, 
		       PARALLEL_FORTRAN_EXT, 
		       text_module(module, mod_stat) );

    close_prettyprint();

    /* free proper effects and cumulated effects 
     Je ne sais pas trop comment le free fonctionne avec statement_effects.
    bc.*/
    /* free_statement_effects( get_rw_effects() );
       free_statement_effects( get_proper_rw_effects() ); */

    reset_rw_effects();
    reset_proper_rw_effects();
    reset_current_module_entity();
    
    set_string_property(PRETTYPRINT_PARALLEL, pp);
    free(pp);

    debug_off();

    return TRUE;
}
