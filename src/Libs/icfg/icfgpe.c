
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

/*static text text_statement_any_effect_type(entity module, int margin, statement stat)
{
  text result = make_text(NIL);
  list l;
  MAPL(l_ips, {
    p_icfgpe_print_stuff ips = (p_icfgpe_print_stuff)STRING(CAR(l_ips));
  }, lp);
}
*/
static text get_any_effects_text_flt(string module_name)
{
  entity module;
  statement module_stat, user_stat = statement_undefined;
  text txt = make_text(NIL);
  
  /* current entity
   */
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  module = get_current_module_entity();

  /* current statement
   */
  set_current_module_statement((statement)db_get_memory_ressource(DBR_CODE, module_name, TRUE));
  module_stat = get_current_module_statement();
  
  /* resources to be printed...
   */
  load_resources(module_name);
  
  debug_on("EFFECTS_DEBUG_LEVEL");
  
  /* init_pretty_print(text_statement_any_effect_type); */

  MERGE_TEXTS(txt, text_module(module, module_stat));
  
  /*close_prettyprint();*/

  debug_off;
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return txt;
}

text get_any_effect_type_text_flt(string module_name, string resource_name, entity e_flt)
{
  text txt;
  add_a_icfgpe_print(resource_name, effects_to_text_func);
  txt = get_any_effects_text_flt(module_name);
  reset_icfgpe_print();
  return txt;
}

text get_text_proper_effects_flt(string module_name, entity e_flt)
{
  text t;
  set_methods_for_rw_effects_prettyprint(module_name);
  t = get_any_effect_type_text(module_name, DBR_PROPER_EFFECTS, string_undefined, TRUE);
  reset_methods_for_effects_prettyprint(module_name);
  return t;
}
