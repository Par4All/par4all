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
#include "patterns.tab.h"
#include <errno.h>


static matchTree patterns_tree = NULL;

static list finalArgType = NIL;
static list curArgType = NIL;

static int gIndexCount = 0;

static basic get_basic_from_pointer(basic inBas, int nbIndex)
{
    basic outBas = basic_undefined;
    type t = basic_pointer(inBas);

    if(!type_variable_p(t))
        return outBas;

    outBas = variable_basic(type_variable(t));

    switch(basic_tag(outBas))
    {
        case is_basic_int:
        case is_basic_float:
        case is_basic_logical:
            if(gIndexCount != nbIndex)
                outBas = basic_undefined;
            break;

        case is_basic_pointer:
            gIndexCount++;
            outBas = get_basic_from_pointer(outBas, nbIndex);
            break;
        default:pips_internal_error("basic_tag %u not supported yet",basic_tag(outBas));
    }

    return outBas;
}

basic get_basic_from_array_ref(reference ref)
{
    basic bas = basic_undefined;
    int nbIndex = gen_length(reference_indices(ref));

    type t = entity_type(reference_variable(ref));

    if(!type_variable_p(t))
        return bas;

    bas = variable_basic(type_variable(ultimate_type(t)));

    gIndexCount = 0;

    switch(basic_tag(bas))
    {
        case is_basic_int:
        case is_basic_float:
        case is_basic_logical:
            break;

        case is_basic_pointer:
            gIndexCount++;
            bas = get_basic_from_pointer(bas, nbIndex);
            break;
        default:pips_internal_error("basic_tag %u not supported yet",basic_tag(bas));
    }

    return bas;
}

void simd_reset_finalArgType()
{
    gen_free_list(finalArgType);
    finalArgType = NIL;

    gen_free_list(curArgType);
    curArgType = NIL;
}

static void simd_fill_curArgType_call(call ca)
{
    FOREACH(EXPRESSION, arg, call_arguments(ca))
    {
        syntax s = expression_syntax(arg);

        switch(syntax_tag(s))
        {
            case is_syntax_call:
                {
                    call c = syntax_call(s);

                    if (call_constant_p(c))
                    {
                        curArgType = CONS(BASIC, make_basic(is_basic_int, 0), curArgType);
                    }
                    else
                        simd_fill_curArgType_call(c);
                } break;

            case is_syntax_reference:
                {
                    basic bas = get_basic_from_array_ref(syntax_reference(s));
                    curArgType = CONS(BASIC, copy_basic(bas), curArgType);
                } break;
            default:pips_internal_error("syntax_tag %u not supported yet",syntax_tag(s));

        }
    }
}

void simd_fill_finalArgType(statement stat)
{
    simd_fill_curArgType_call(statement_call(stat));
    finalArgType = gen_copy_seq(curArgType);
}

void simd_fill_curArgType(statement stat)
{
    simd_fill_curArgType_call(statement_call(stat));
}

bool simd_check_argType()
{
    list pCurBas = curArgType;

    FOREACH(BASIC, finBas, finalArgType)
	{
		basic curBas = BASIC(CAR(pCurBas));

		if((!(((basic_int_p(finBas)) &&  (basic_int(finBas) == 0)) ||
						((basic_int_p(curBas)) &&  (basic_int(curBas) == 0)))) &&
				!basic_equal_p(curBas, finBas))
		{
			return false;
		}

		else if(((basic_int_p(finBas)) &&  (basic_int(finBas) == 0)))
		{
			basic_tag(finBas) = basic_tag(curBas);

			switch(basic_tag(curBas))
			{
				case is_basic_int: basic_int(finBas) = basic_int(curBas); break;
				case is_basic_float: basic_float(finBas) = basic_float(curBas); break;
				case is_basic_logical: basic_logical(finBas) = basic_logical(curBas); break;
				default:pips_internal_error("basic_tag %u not supported yet",basic_tag(curBas));
			}
		}

		pCurBas = CDR(pCurBas);
		if(ENDP(pCurBas) )
			return false;

	}

    gen_free_list(curArgType);
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
    if (!top_level_entity_p(call_function(c)) || 
            call_constant_p(c))
        return matchTree_undefined; /* no match */

    t = select_tree_branch(t, get_operator_id(call_function(c)));
    if (t == matchTree_undefined)
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
                    basic bas = get_basic_from_array_ref(syntax_reference(s));
                    if(bas == basic_undefined)
                    {
                        return matchTree_undefined;
                    }
                    else
                    {
                        t = select_tree_branch(t, REFERENCE_TOK);
                        *args = CONS(EXPRESSION, arg, *args);
                    }
                    break;
                }

            case is_syntax_range:
            default:
                return matchTree_undefined; /* unexpected token !! -> no match */
        }

        if (t == matchTree_undefined)
            return matchTree_undefined;
    }
        

    return t;
}

/* merge the 2 lists. Warning: list gets reversed. */
static list merge_lists(list l, list format)
{
    list res = NIL;

    /* merge according to the format specifier list */
    for( ; format != NIL; format = CDR(format))
    {
        patternArg param = PATTERNARG(CAR(format));

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
    matchTree t;
    list args = NULL;
    list matches = NIL;
    list i;

    if (!statement_call_p(s))
        return NIL;

    /* find the matching patterns */
    t = match_call(statement_call(s), patterns_tree, &args);
    if (t == matchTree_undefined)
    {
        gen_free_list(args);
        return NIL;
    }

    /* build the matches */
    for(i = matchTree_patterns(t); i != NIL; i = CDR(i)) 
    {
        patternx p = PATTERNX(CAR(i));
        match m = make_match(patternx_class(p), 
                merge_lists(args, patternx_args(p)));

        matches = CONS(MATCH, m, matches);
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

void patterns_yyparse();

void init_tree_patterns()
{
    extern FILE * patterns_yyin;

    if (patterns_tree == NULL) /* never initialized */
    {
        patterns_tree = make_tree();
        patterns_yyin=fopen_config("patterns.def","SIMD_PATTERN_FILE",NULL);
        patterns_yyparse();
        fclose(patterns_yyin);
    }
}
