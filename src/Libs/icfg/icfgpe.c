
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

void reset_icfgpe_print()
{
  gen_map(free, lp);
  gen_free_list(lp);
  lp = NIL;
}

void add_a_icfgpe_print(string resource_name, get_text_function gt)
{
  p_icfgpe_print_stuff ips = (p_icfgpe_print_stuff)malloc(sizeof(icfgpe_print_stuff));
  ips->name = resource_name;
  ips->resource = gen_chunk_undefined;
  ips->get_text = gt;
  
  lp = CONS(STRING, (char *)ips, lp);
}

text get_any_effect_type_text_flt(string module_name, string resource_name, entity e_flt)
{
  text txt;
  
  /*txt = get_any_effects_text(module_name, TRUE);*/
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
