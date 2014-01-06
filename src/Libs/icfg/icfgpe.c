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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "control.h"      /* CONTROL_MAP is defined there */
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"  /* get_bool_property */
#include "misc.h"
#include "database.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "resources.h"
#include "semantics.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "pipsdbm.h"      /* DB_PUT_FILE_RESOURCE is defined there */
#include "text-util.h"
#include "icfg.h"

list /* of effect */ effects_filter(list l_effs)
{
  entity e_flt = FindEntity("MAIN", "KMAX");
  if (!entity_undefined_p(e_flt)) {
    list l_flt = NIL;
    MAPL(l, {
      effect eff = EFFECT(CAR(l));
      action ac = effect_action(eff);
      reference ref = effect_any_reference(eff);
      entity ent = reference_variable(ref);
      if (entities_may_conflict_p(e_flt, ent) && !action_read_p(ac))
	l_flt = CONS(EFFECT, eff, l_flt);
    }, l_effs);
    return l_flt;
  } else
    return l_effs;
}

text get_text_proper_effects_flt(const char* module_name)
{
  text r = make_text(NIL);
  entity module;
  statement module_stat;
  statement s;
  gen_chunk * m = (gen_chunk *) db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true);
  
  /* current entity
   */
  set_current_module_entity( local_name_to_top_level_entity(module_name));
  module = get_current_module_entity();
  
  /* current statement
   */
  set_current_module_statement((statement) db_get_memory_resource
			       (DBR_CODE, module_name, true));
  module_stat = get_current_module_statement();
  
  {
    instruction i = statement_instruction(module_stat);
    if instruction_block_p(i) {
      list objs = instruction_block(i);
      for (; objs != NIL; objs = CDR(objs)) {
	statement s = STATEMENT(CAR(objs));
	list l_effs = effects_effects(apply_statement_effects(m, s));
	list l_effs_flt = effects_filter(l_effs);
	if (l_effs_flt != NIL) {
	  text s_text = text_statement(entity_undefined, 0, s);
	  text t = simple_rw_effects_to_text(l_effs_flt);
	  text_sentences(r) = gen_nconc(text_sentences(r), text_sentences(t));
	  text_sentences(t) = NIL;
	  free_text(t);
	  text_sentences(r) = gen_nconc(text_sentences(r), text_sentences(s_text));
	  text_sentences(s_text) = NIL;
	  free_text(s_text);
	}
      } 
    }
  }
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return r;
}
















