/* Created by B. Apvrille, april 11th, 1994 */
/* functions related to types effects and effect */

#include <stdio.h>

#include "linear.h"

#include "genC.h"

#include "ri.h"

/* #include"mapping.h" */
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
   the argument (formal parameter) of the module. Note that the return
   variable of a function is also considered here as a formal
   parameter. */
bool
statement_has_a_module_formal_argument_write_effect_p(statement s,
                                                      entity module,
                                                      statement_mapping effects_list_map)
{
   bool write_effect_on_a_module_argument_found = FALSE;
   list effects_list = (list) GET_STATEMENT_MAPPING(effects_list_map, s);

   MAP(EFFECT, an_effect,
       {
          entity a_variable = reference_variable(effect_reference(an_effect));
          
          if (action_write_p(effect_action(an_effect))
              && (variable_return_p(a_variable)
		  || variable_is_a_module_formal_parameter_p(a_variable,
							     module))) {
	      write_effect_on_a_module_argument_found = TRUE;
             break;
          }
       },
       effects_list);

   return write_effect_on_a_module_argument_found;

}
