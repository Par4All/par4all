/*
 * $Id$
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "properties.h"  /* get_bool_property */
#include "misc.h"
#include "database.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "resources.h"
#include "semantics.h"
#include "prettyprint.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "pipsdbm.h"      /* DB_PUT_FILE_RESOURCE is defined there */
#include "text-util.h"
#include "icfg.h"

static bool found_filter = FALSE;

static text 
text_block(
    entity module,
    string label,
    int margin,
    list objs,
    int n)
{
    text r = make_text(NIL);
    list pbeg, pend ;

    pend = NIL;

    if (ENDP(objs) && !get_bool_property("PRETTYPRINT_EMPTY_BLOCKS")) {
	return(r) ;
    }


    if(!empty_local_label_name_p(label)) {
	pips_internal_error("Illegal label \"%s\". "
			    "Blocks cannot carry a label\n",
			    label);
    }
    
    if (get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
	get_bool_property("PRETTYPRINT_BLOCKS")) {
	unformatted u;
	
	if (get_bool_property("PRETTYPRINT_FOR_FORESYS")){
	    ADD_SENTENCE_TO_TEXT(r, make_sentence(is_sentence_formatted, 
						  strdup("C$BB\n")));
	}
	else {
	    pbeg = CHAIN_SWORD(NIL, "BEGIN BLOCK");
	    pend = CHAIN_SWORD(NIL, "END BLOCK");
	    
	    u = make_unformatted(strdup("C"), n, margin, pbeg);
	    ADD_SENTENCE_TO_TEXT(r, 
				 make_sentence(is_sentence_unformatted, u));
	}
    }

    for (; objs != NIL; objs = CDR(objs)) {
	statement s = STATEMENT(CAR(objs));

	text t = text_statement(module, margin, s);
	/**********written by Dat************/
	if (found_filter) {
	  text_sentences(r) = 
	    gen_nconc(text_sentences(r), text_sentences(t));
	}
	text_sentences(t) = NIL;
	free_text(t);
    }

    if (!get_bool_property("PRETTYPRINT_FOR_FORESYS") &&
	(get_bool_property("PRETTYPRINT_ALL_EFFECTS") ||
	 get_bool_property("PRETTYPRINT_BLOCKS"))) 
    {
	unformatted u = make_unformatted(strdup("C"), n, margin, pend);
	ADD_SENTENCE_TO_TEXT(r, 
			     make_sentence(is_sentence_unformatted, u));
    }
    return r;
}

typedef struct
{
  string name;
  gen_chunk * resource;
  get_text_function get_text;
} icfgpe_print_stuff, * p_icfgpe_print_stuff;

static p_icfgpe_print_stuff ips = NULL;

void create_ips(string module_name, string resource_name, get_text_function gt)
{
  ips = (p_icfgpe_print_stuff)malloc(sizeof(icfgpe_print_stuff));
  ips->name = resource_name;
  ips->resource = (gen_chunk*) db_get_memory_resource(resource_name, module_name, TRUE);
  ips->get_text = gt;
}

static list
load_list_icfg(statement_effects m, statement s) {
  return effects_effects(apply_statement_effects(m, s));
}

list /* of effect */ effects_filter(list l_effs, entity e_flt)
{
  list l_flt = NIL;
  MAPL(l, {
    effect eff = EFFECT(CAR(l));
    action ac = effect_action(eff);
    reference ref = effect_reference(eff);
    entity ent = reference_variable(ref);
    if (entity_conflict_p(e_flt, ent) && !action_read_p(ac))
      l_flt = CONS(EFFECT, eff, l_flt);
  }, l_effs);
  return l_flt;
}

static text
resource_text_flt(entity module, int margin, statement stat)
{
  list l_eff = load_list_icfg(ips->resource, stat);
  entity e_flt = global_name_to_entity("MAIN", "KMAX");
  list l_eff_flt = effects_filter(l_eff, e_flt);
  text l_eff_text = (*(ips->get_text))(l_eff_flt);
  gen_free_list(l_eff_flt);
  return l_eff_text;
}

/*text text_statement_flt(entity module, int margin, statement stmt)
{
  instruction i = statement_instruction(stmt);
  text r = make_text(NIL);
  text temp = text_instruction(module, margin, i, statement_number(stmt));
  found_filter = FALSE;
  if (!ENDP(text_sentences(temp))) {
    text t = init_text_statement(module, margin, stmt);
    if (!ENDP(text_sentences(t))) {
      MERGE_TEXTS(r, t);
      MERGE_TEXTS(r, temp);
      found_filter = TRUE;
    } else {
      MERGE_TEXTS(r, temp);
    }
  } else {
    free(temp);
  }
  return r;
}*/

static text
get_any_effects_text_flt(string module_name)
{
  entity module;
  statement module_stat;
  text txt = make_text(NIL);
  
  /* current entity
   */
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  module = get_current_module_entity();

  /* current statement
   */
  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name, TRUE));
  module_stat = get_current_module_statement();
  
  debug_on("EFFECTS_DEBUG_LEVEL");
  
  init_prettyprint(resource_text_flt);

  MERGE_TEXTS(txt, text_statement(module, 0, module_stat));
  
  close_prettyprint();

  debug_off();
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return txt;
}

text get_text_proper_effects_flt(string module_name)
{
  text txt;
  set_methods_for_rw_effects_prettyprint(module_name);
  create_ips(module_name, DBR_PROPER_EFFECTS, effects_to_text_func);
  txt = get_any_effects_text_flt(module_name);
  free(ips);
  ips = NULL;
  reset_methods_for_effects_prettyprint(module_name);
  return txt;
}




