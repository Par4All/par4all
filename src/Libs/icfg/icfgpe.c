
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

typedef struct
{
  string name;
  gen_chunk * resource;
  get_text_function get_text;
} icfgpe_print_stuff, * p_icfgpe_print_stuff;

static list lp = NIL;


static
void reset_icfgpe_print()
{
  gen_map(free, lp);
  gen_free_list(lp);
  lp = NIL;
}

static
void add_a_icfgpe_print(string resource_name, get_text_function gt)
{
  p_icfgpe_print_stuff ips = (p_icfgpe_print_stuff)malloc(sizeof(icfgpe_print_stuff));
  ips->name = resource_name;
  ips->resource = gen_chunk_undefined;
  ips->get_text = gt;
  
  lp = CONS(STRING, (char *)ips, lp);
}

static list
load_list_icfg(statement_effects m, statement s) {
  return effects_effects(apply_statement_effects(m, s));
}

list effects_filter(list l_effs, entity e_flt)
{
  list l_flt = NIL;
  MAPL(ce, {
    effect eff = EFFECT(CAR(ce));
    action ac = effect_action(eff);
    reference ref = effect_reference(eff);
    if (entity_conflict_p(e_flt, reference_variable(ref)) && !action_read_p(ac))
      l_flt = CONS(EFFECT, eff, l_flt);
    free(t);
  }, l_effs);
  return l_flt;
}

static text
resource_text_flt(entity module, int margin, statement stat, p_icfgpe_print_stuff ips)
{
  list l_eff = load_list_icfg(ips->resource, stat);
  entity e_flt = FindOrCreateEntity(TOP_LEVEL_MODULE, "KMAX");
  list l_eff_flt = effects_filter(l_eff, e_flt);
  text l_eff_text = (*(ips->get_text))(l_eff_flt);
  gen_free_list(l_eff_flt);
  return l_eff_text;
}

static text text_statement_any_effect_type_flt(entity module, int margin, statement stat)
{
  text result = make_text(NIL);
  MAPL(l, {
    p_icfgpe_print_stuff ips = (p_icfgpe_print_stuff) STRING(CAR(l));
    MERGE_TEXTS(result, resource_text_flt(module, margin, stat, ips));
  }, lp);
  return result;
}

static void
load_resources_icfg(string module_name)
{
  MAPL(l, {
    p_icfgpe_print_stuff ips = (p_icfgpe_print_stuff) STRING(CAR(l));
    ips->resource = (gen_chunk*) db_get_memory_resource(ips->name, module_name, TRUE);
  }, lp);
}

/*text text_statement_flt(entity module, int margin, statement stmt)
{
  instruction i = statement_instruction(stmt);
  text r = make_text(NIL);
  text temp;
  string label = entity_local_name(statement_label(stmt)) + strlen(LABEL_PREFIX);
  if (same_string_p(label, RETURN_LABEL_NAME))
    temp = make_text(NIL);
  else
    temp = text_instruction(module, label, margin, i, statement_number(stmt));
  if (!ENDP(text_sentences(temp))) {
    text t = init_text_statement(module, margin, stmt);
    if (!ENDP(text_sentences(t))) {
      MERGE_TEXTS(r, t);
      MERGE_TEXTS(r, temp);
    } else {
      MERGE_TEXTS(r, temp);
    }
  } else
    free_text(temp);
  return r;
}*/

static text get_any_effects_text_flt(string module_name)
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
  
  /* resources to be printed...
   */
  load_resources_icfg(module_name);
  
  debug_on("EFFECTS_DEBUG_LEVEL");
  
  init_prettyprint(text_statement_any_effect_type_flt);

  MERGE_TEXTS(txt, text_statement(module, 0, module_stat));
  
  close_prettyprint();

  debug_off();
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return txt;
}

text get_any_effect_type_text_flt(string module_name, string resource_name)
{
  text txt;
  add_a_icfgpe_print(resource_name, effects_to_text_func);
  txt = get_any_effects_text_flt(module_name);
  reset_icfgpe_print();
  return txt;
}

text get_text_proper_effects_flt(string module_name)
{
  text t;
  set_methods_for_rw_effects_prettyprint(module_name);
  t = get_any_effect_type_text_flt(module_name, DBR_PROPER_EFFECTS);
  reset_methods_for_effects_prettyprint(module_name);
  return t;
}
