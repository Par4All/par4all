
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

text get_text_proper_effects_flt(string module_name, entity e_flt)
{
  text t;
  set_is_user_view_p(FALSE);
  set_methods_for_rw_effects_prettyprint(module_name);
  t = get_any_effect_type_text(module_name, DBR_PROPER_EFFECTS, string_undefined, TRUE);
  reset_methods_for_effects_prettyprint(module_name);
  return t;
}
