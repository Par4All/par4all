/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "pipsdbm.h"
#include "prettyprint.h"
#include "preprocessor.h"
#include "expressions.h"

#include "top-level.h"
#include "resources.h"

#include "transformations.h"


/* returns a list of all entities which:
 * - are concerned with cumulated effects (cfx) of the loop_body
 * - and are member of loop_locals(lp)
 * makes sure the loop index is in the list
 */
static list real_loop_locals(loop lp, effects cfx)
{
    list rll= NIL;

    MAPL(ce, 
     { 
	effect 
	    eff = EFFECT(CAR(ce));
	entity 
	    ent = reference_variable(effect_any_reference(eff));

	if (!entity_in_list_p(ent, rll)
	    && entity_in_list_p(ent,loop_locals(lp)) ) 
	{
	    /* ent is a real loop local */
	    debug(7, "real_loop_locals", "real loop local: %s\n",
		  entity_local_name(ent));
	    rll = CONS(ENTITY, ent, rll);
	}
    }, effects_effects(cfx));

    if( !entity_in_list_p(loop_index(lp), rll) ) 
    {
	rll= CONS(ENTITY, loop_index(lp), rll);
    }
    return(rll);
}

/*
 * recursively concatenates all real loop locals of all enclosed loops.
 * filters redondant entities
 */
static list all_enclosed_scope_variables(statement stmt)
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
      case is_instruction_whileloop : {
	whileloop lp= instruction_whileloop(instr);
	ent_l= all_enclosed_scope_variables(whileloop_body(lp));
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
	pips_internal_error("Bad instruction tag");
    }
    return(ent_l);
}

/* lp_stt must be a loop statement */
static text text_microtasked_loop(__attribute__((unused)) entity module,__attribute__((unused))  int margin, statement lp_stt)
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
	/* if (!ENDP(reference_indices(effect_any_reference(eff)))) */

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
	entity ent = reference_variable(effect_any_reference(eff));

/*	if(ENDP(ce)) user_log("ce is NIL !!\n");
	if(ENDP(CDR(ce))) user_log("CDR(ce) is NIL\n");
	else user_log("CDR(ce) is *not* NIL\n");
 */
	/* What about arrays? nothing special?? */
	/* if (!ENDP(reference_indices(effect_any_reference(eff)))) */

	/*
	 * ent_l added, in order to use the newly computed << good >> effects, 
	 * and to warrant the fact that no variables should be declared
	 * at the same time private and shared.
	 * same_entity_p added also, just in case.
	 *
	 * FC 28/09/93
	 */
	if ((!entity_in_list_p(ent, loop_locals(lp))) &&
	    (!entity_in_list_p(ent, ent_l)) &&
	    (!same_entity_p(ent, loop_index(lp))) &&
	    (!entity_in_list_p(ent, lp_shared)))
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
static text text_vectorized_loop(__attribute__((unused)) entity module,__attribute__((unused))  int margin,__attribute__((unused))  statement lp_stt)
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


static text text_cray(entity module, int margin, statement stat)
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
    entity module = module_name_to_entity(mod_name);
    statement mod_stat = statement_undefined;

    /* push prettyprint style */
    string pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
    set_string_property(PRETTYPRINT_PARALLEL, "cray");

    set_current_module_entity(module);
    mod_stat = (statement)
	db_get_memory_resource(DBR_PARALLELIZED_CODE, mod_name, true);
    
    /* We need to recompute proper effects and cumulated effects */
    init_proper_rw_effects();
    rproper_effects_of_statement(mod_stat);

    init_rw_effects();
    rcumulated_effects_of_statement(mod_stat);

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

    return true;
}
