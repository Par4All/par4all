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

