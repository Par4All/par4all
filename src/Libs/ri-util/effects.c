/* Created by B. Apvrille, april 11th, 1994 */
/* functions related to types effects and effect */

#include "genC.h"

#include "ri.h"

#include"mapping.h"
#include "misc.h"

#include "ri-util.h"

/* ---------------------------------------------------------------------- */
/* list-effects conversion functions                                      */
/* ---------------------------------------------------------------------- */

effects list_to_effects(l_eff)
list l_eff;
{
    effects res = make_effects(l_eff);
    return res;
}

list effects_to_list(efs)
effects efs;
{
    list l_res = effects_effects(efs);
    return l_res;
}

statement_mapping listmap_to_effectsmap(l_map)
statement_mapping l_map;
{
    statement_mapping efs_map = MAKE_STATEMENT_MAPPING();
    
    STATEMENT_MAPPING_MAP(s,val,{
	hash_put((hash_table) efs_map, (char *) s, (char *) list_to_effects((list) val));
    }, l_map);

    return efs_map;
}

statement_mapping effectsmap_to_listmap(efs_map)
statement_mapping efs_map;
{
    statement_mapping l_map = MAKE_STATEMENT_MAPPING();
    
    STATEMENT_MAPPING_MAP(s,val,{
	hash_put((hash_table) l_map, (char *) s, (char *) effects_to_list((effects) val));
    }, efs_map);

    return l_map;
}



/* Return TRUE if the statement has a write effect on at least one of
   the argument of the module. */
bool
statement_write_argument_of_module_effect(statement s,
                                          entity module,
                                          statement_mapping effects_list_map)
{
   bool write_effect_on_a_module_argument_found = FALSE;
   list effects_list = (list) GET_STATEMENT_MAPPING(effects_list_map, s);

   MAP(EFFECT, an_effect,
       {
          entity a_variable = reference_variable(effect_reference(an_effect));
          
          if (action_write_p(effect_action(an_effect))
              && variable_in_module_p(a_variable, module)) {
             write_effect_on_a_module_argument_found = TRUE;
             break;
          }
       },
       effects_list);

   return write_effect_on_a_module_argument_found;

}


/* Return true if a statement has an I/O effect in the effects
   list. */
bool
statement_io_effect_p(statement_mapping effects_list_map,
                      statement s)
{
   bool io_effect_found = FALSE;
   list effects_list = (list) GET_STATEMENT_MAPPING(effects_list_map, s);
   /* If there is an I/O effects, the following entity should
      exist. If it does not exist, statement_io_effect_p() will return
      FALSE anyway. */
   entity private_io_entity =
      global_name_to_entity(TOP_LEVEL_MODULE_NAME,
                            IO_EFFECTS_ARRAY_NAME);

   MAP(EFFECT, an_effect,
       {
          entity a_touched_variable =
             reference_variable(effect_reference(an_effect));
          if (a_touched_variable == private_io_entity) {
             io_effect_found = TRUE;
             break;
          }
       },
       effects_list);

   return io_effect_found;
}
