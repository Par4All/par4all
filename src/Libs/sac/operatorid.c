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
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "sac.h"
#include "patterns.h"
#include "pipsdbm.h"

static operator_id_tree mappings = NULL;
void set_simd_operator_mappings(void * m)
{
    pips_assert("not already set",mappings==NULL);
    mappings=m;
}
void reset_simd_operator_mappings()
{
    pips_assert("already set",mappings);
    mappings=NULL;
}

/*
 * we manipulate tokens from
 * either the `ri'
 * or from the pattern lexer
 * binding is done here
 */

typedef struct {
    char* name;
    int id;
} oper_id_mapping;

static oper_id_mapping operators[] =
{
    { ASSIGN_OPERATOR_NAME,            ASSIGN_OPERATOR_TOK },
    { PLUS_OPERATOR_NAME,              PLUS_OPERATOR_TOK },
    { PLUS_C_OPERATOR_NAME,            PLUS_OPERATOR_TOK },
    { MINUS_OPERATOR_NAME,             MINUS_OPERATOR_TOK },
    { COS_OPERATOR_NAME,               COS_OPERATOR_TOK },
    { SIN_OPERATOR_NAME,               SIN_OPERATOR_TOK },
    { UNARY_MINUS_OPERATOR_NAME,       UNARY_MINUS_OPERATOR_TOK },
    { MULTIPLY_OPERATOR_NAME,          MULTIPLY_OPERATOR_TOK },
    { DIVIDE_OPERATOR_NAME,            DIVIDE_OPERATOR_TOK },
    { INVERSE_OPERATOR_NAME,           INVERSE_OPERATOR_TOK },
    { POWER_OPERATOR_NAME,             POWER_OPERATOR_TOK },
    { MODULO_OPERATOR_NAME,            MODULO_OPERATOR_TOK },
    { MIN_OPERATOR_NAME,               MIN_OPERATOR_TOK },
    { MIN0_OPERATOR_NAME,              MIN0_OPERATOR_TOK },
    { AMIN1_OPERATOR_NAME,             AMIN1_OPERATOR_TOK },
    { DMIN1_OPERATOR_NAME,             DMIN1_OPERATOR_TOK },
    { MAX_OPERATOR_NAME,               MAX_OPERATOR_TOK },
    { MAX0_OPERATOR_NAME,              MAX0_OPERATOR_TOK },
    { AMAX1_OPERATOR_NAME,             AMAX1_OPERATOR_TOK },
    { DMAX1_OPERATOR_NAME,             DMAX1_OPERATOR_TOK },
    { ABS_OPERATOR_NAME,               ABS_OPERATOR_TOK },
    { IABS_OPERATOR_NAME,              IABS_OPERATOR_TOK },
    { DABS_OPERATOR_NAME,              DABS_OPERATOR_TOK },
    { CABS_OPERATOR_NAME,              CABS_OPERATOR_TOK },

    { AND_OPERATOR_NAME,               AND_OPERATOR_TOK },
    { OR_OPERATOR_NAME,                OR_OPERATOR_TOK },
    { NOT_OPERATOR_NAME,               NOT_OPERATOR_TOK },
    { NON_EQUAL_OPERATOR_NAME,         NON_EQUAL_OPERATOR_TOK },
    { EQUIV_OPERATOR_NAME,             EQUIV_OPERATOR_TOK },
    { NON_EQUIV_OPERATOR_NAME,         NON_EQUIV_OPERATOR_TOK },

    { TRUE_OPERATOR_NAME,              TRUE_OPERATOR_TOK },
    { FALSE_OPERATOR_NAME,             FALSE_OPERATOR_TOK },

    { C_GREATER_OR_EQUAL_OPERATOR_NAME,  GREATER_OR_EQUAL_OPERATOR_TOK },
    { C_GREATER_THAN_OPERATOR_NAME,      GREATER_THAN_OPERATOR_TOK },
    { C_LESS_OR_EQUAL_OPERATOR_NAME,     LESS_OR_EQUAL_OPERATOR_TOK },
    { C_LESS_THAN_OPERATOR_NAME,         LESS_THAN_OPERATOR_TOK },
    { C_EQUAL_OPERATOR_NAME,             EQUAL_OPERATOR_TOK },
    { CONDITIONAL_OPERATOR_NAME,         PHI_TOK },

    { "__PIPS_SAC_MULADD",         	 MULADD_OPERATOR_TOK },

    { NULL,                              UNKNOWN_TOK }
};


static void insert_mapping(oper_id_mapping* item)
{
    char * s;
    operator_id_tree t;

    t = mappings;
    for(s = item->name; *s != 0; s++)
    {
        operator_id_tree next;
        intptr_t c = *s;

        next = (operator_id_tree)hash_get(operator_id_tree_sons(t), (void*)c);
        if (next == HASH_UNDEFINED_VALUE)
        {
            next = make_operator_id_tree(UNKNOWN_TOK,hash_table_make(hash_int,HASH_DEFAULT_SIZE));
            hash_put(operator_id_tree_sons(t), (void *)c, (void*)next);
        }

        t = next;
    }

    if (operator_id_tree_id(t) != UNKNOWN_TOK)
        pips_user_warning("overwriting previous mapping...\n");

    operator_id_tree_id(t) = item->id;
}

static int do_get_operator_id(const char* ename)
{
    operator_id_tree t = mappings;
    for(const char *s = ename; *s != 0; s++)
    {
        operator_id_tree next;
        intptr_t c = *s;

        next = (operator_id_tree)hash_get(operator_id_tree_sons(t), (void*)c);
        if (next == HASH_UNDEFINED_VALUE)
        {
            return UNKNOWN_TOK;
        }
        t = next;
    }
    return operator_id_tree_id(t);
}

int get_operator_id(entity e)
{
    const char* ename = entity_local_name(e);
    int res = do_get_operator_id(ename);
    if(res == UNKNOWN_TOK )
    {
        /* retry with uppercase version, cos -> COS :) */
        ename = strupper(strdup(ename),ename);
        res = do_get_operator_id(ename);
    }
    return res;
}

bool simd_operator_mappings(__attribute__((unused)) char * module_name)
{
    /* create a new operator id */
    operator_id_tree m= make_operator_id_tree(UNKNOWN_TOK,hash_table_make(hash_int,HASH_DEFAULT_SIZE));
    set_simd_operator_mappings(m);

    for(size_t i=0; operators[i].name != NULL; i++)
        insert_mapping(&operators[i]);

    /* put it in pipsdbm */
    DB_PUT_MEMORY_RESOURCE(DBR_SIMD_OPERATOR_MAPPINGS,"",m);

    reset_simd_operator_mappings();
    return true;
}

