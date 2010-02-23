/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"
#include "properties.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac.h"
#include "patterns.h"
#include <errno.h>


static matchTree patterns_tree = NULL;
void set_simd_treematch(matchTree t)
{
    pips_assert("tree match not already set",patterns_tree==NULL);
    patterns_tree=t;
}

void reset_simd_treematch()
{
    pips_assert("tree match already set",patterns_tree!=NULL);
    patterns_tree=NULL;
}

static list finalArgType = NIL;
static list curArgType = NIL;

void simd_reset_finalArgType()
{
    gen_free_list(finalArgType);
    finalArgType = NIL;

    gen_free_list(curArgType);
    curArgType = NIL;
}
static list simd_fill_curArgType_call(call ca)
{
    list c=NIL;
    FOREACH(EXPRESSION, arg, call_arguments(ca))
        c= CONS(BASIC,basic_of_expression(arg),c);
    return c;
}

void simd_fill_finalArgType(statement stat)
{
    finalArgType=simd_fill_curArgType_call(statement_call(stat));
}

void simd_fill_curArgType(statement stat)
{
    curArgType=simd_fill_curArgType_call(statement_call(stat));
}

bool simd_check_argType()
{
    list pCurBas = curArgType;

    FOREACH(BASIC, finBas, finalArgType)
	{
		if(ENDP(pCurBas) )
			return false;
		basic curBas = BASIC(CAR(pCurBas));
        if(!basic_equal_p(curBas, finBas))
            return false;
		pCurBas = CDR(pCurBas);
	}

    gen_full_free_list(curArgType);
    curArgType = NIL;

    return true;
}


static matchTree make_tree()
{
    matchTreeSons sons = make_matchTreeSons();
    matchTree n = make_matchTree(NIL, sons);
    return n;
}

static void insert_tree_branch(matchTree t, int token, matchTree n)
{
    extend_matchTreeSons(matchTree_sons(t), token, n);
}

static matchTree select_tree_branch(matchTree t, int token)
{
    if (bound_matchTreeSons_p(matchTree_sons(t), token))
        return apply_matchTreeSons(matchTree_sons(t), token);
    else
        return matchTree_undefined;
}

/* Warning: list of arguments is built in reversed order
 * (the head is in fact the last argument) */
static matchTree match_call(call c, matchTree t, list *args)
{
    if (!top_level_entity_p(call_function(c)) || call_constant_p(c))
        return matchTree_undefined; /* no match */

    t = select_tree_branch(t, get_operator_id(call_function(c)));
    if (matchTree_undefined_p(t))
        return matchTree_undefined; /* no match */

    FOREACH(EXPRESSION, arg, call_arguments(c))
    {
        syntax s = expression_syntax(arg);
        switch(syntax_tag(s))
        {
            case is_syntax_call:
                {
                    call c = syntax_call(s);
                    if (call_constant_p(c))
                    {
                        t = select_tree_branch(t, CONSTANT_TOK);
                        *args = CONS(EXPRESSION, arg, *args);
                    }
                    else
                        t = match_call(c, t, args);
                    break;
                }

            case is_syntax_reference:
                {
                    basic bas = basic_of_reference(syntax_reference(s));
                    if(bas == basic_undefined)
                    {
                        return matchTree_undefined;
                    }
                    else
                    {
                        t = select_tree_branch(t, REFERENCE_TOK);
                        *args = CONS(EXPRESSION, arg, *args);
                        free_basic(bas);
                    }
                    break;
                }

            case is_syntax_range:
            default:
                return matchTree_undefined; /* unexpected token !! -> no match */
        }

        if (matchTree_undefined_p(t) )
            return matchTree_undefined;
    }
        

    return t;
}

/* merge the 2 lists. Warning: list gets reversed. */
static list merge_lists(list l, list format)
{
    list res = NIL;

    /* merge according to the format specifier list */
    FOREACH(PATTERNARG,param,format)
    {
        if (patternArg_dynamic_p(param))
        {
            if (l != NIL)
            {
                res = CONS(EXPRESSION, EXPRESSION(CAR(l)), res);
                l = CDR(l);
            }
            else
                pips_user_warning("Trying to use a malformed rule. Ignoring missing parameter...\n");
        }
        else
        {
            expression e = make_integer_constant_expression(patternArg_static(param));

            res = CONS(EXPRESSION, e, res);
        }
    }

    /* append remaining arguments, if any */
    for( ; l != NIL; l = CDR(l) )
        res = CONS(EXPRESSION, EXPRESSION(CAR(l)), res);

    return res;
}

/* return a list of matching statements */
list match_statement(statement s)
{
    list matches = NIL;

    if (statement_call_p(s))
    {

        /* find the matching patterns */
        list args = NIL;
        matchTree t = match_call(statement_call(s), patterns_tree, &args);
        if (matchTree_undefined_p(t))
        {
            gen_free_list(args);
            return NIL;
        }

        /* build the matches */
        FOREACH(PATTERNX, p ,matchTree_patterns(t))
        {
            match m = make_match(
                    patternx_class(p),
                    merge_lists(args, patternx_args(p))
                    );
            matches = CONS(MATCH, m, matches);
        }
    }

    return matches;
}

void insert_opcodeClass(char * s, int nbArgs, list opcodes)
{
    //Add the class and name in the map
    make_opcodeClass(s, nbArgs, opcodes);
}

opcodeClass get_opcodeClass(char * s)
{
    return gen_find_opcodeClass(s);
}

void insert_pattern(char * s, list tokens, list args)
{
    opcodeClass c = get_opcodeClass(s);
    patternx p;
    matchTree m = patterns_tree;

    if (c == opcodeClass_undefined)
    {
        pips_user_warning("defining pattern for an undefined opcodeClass (%s).",s);
        return;
    }

    p = make_patternx(c, args);

    for( ; tokens != NIL; tokens = CDR(tokens) )
    {
        int token = INT(CAR(tokens));
        matchTree next = select_tree_branch(m, token);

        if (next == matchTree_undefined)
        {
            /* no such branch -> create a new branch */
            next = make_tree();
            insert_tree_branch(m, token, next);
        }

        m = next;
    }

    matchTree_patterns(m) = CONS(PATTERNX, p, matchTree_patterns(m));
}
extern void patterns_yyparse();
extern FILE * patterns_yyin;

bool simd_treematcher(__attribute__((unused)) char * module_name)
{
    /* create a new tree match */
    matchTree treematch = make_tree();
    set_simd_treematch(treematch);

    /* fill it */
    patterns_yyin=fopen_config("patterns.def","SIMD_PATTERN_FILE",NULL);
    patterns_yyparse();
    fclose(patterns_yyin);

    /* put it in pipsdbm */
    DB_PUT_MEMORY_RESOURCE(DBR_SIMD_TREEMATCH,"",treematch);

    /* clean up */
    reset_simd_treematch();
    return true;
}
