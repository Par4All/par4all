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
/*
 * Perform forward substitution when possible.
 *
 * The main purpose of such a transformation is to undo manual
 * optimizations targeted to some particular architecture...
 * 
 * What kind of interface should be available?
 * global? per loop? 
 *
 * basically there should be some common stuff with partial evaluation,
 * as suggested by CA, and I agree, but as I looked at the implementation
 * in the other file, I cannot really guess how to merge and parametrize
 * both stuff as one... FC. 
 *
 * This information is only implemented to forward substitution within 
 * a sequence. Things could be performed at the control flow graph level.
 *
 * An important issue is to only perform the substitution only if correct.
 * Thus conversions are inserted and if none is available, the propagation
 * and substitution are not performed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;


#include "graph.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"

#include "expressions-local.h"
#include "expressions.h"

#include "ricedg.h"

/* structure to hold a substitution to be performed forward.
 */
typedef struct
{
    statement source; /* the statement where the definition was found. */
    reference ref;       /* maybe could be a reference to allow arrays? SG:done*/
    expression val;   /* the result of the substitution. */
}
    t_substitution, * p_substitution;

/* newgen-looking make/free
 */
static p_substitution 
make_substitution(statement source, reference ref, expression val)
{
    p_substitution subs;
    basic bval = basic_of_expression(val),
          bref = basic_of_reference(ref);

    pips_assert("defined basics", !(basic_undefined_p(bval) || basic_undefined_p(bref)));

    subs = (p_substitution) malloc(sizeof(t_substitution));
    pips_assert("malloc ok", subs);
    subs->source = source;
    subs->ref = ref;
    if (basic_equal_p(bval, bref))
        subs->val = copy_expression(val);
    else
    {
        entity conv = basic_to_generic_conversion(bref);
        if (entity_undefined_p(conv))
        {
            pips_user_warning("no conversion function...");
            free(subs); 
            return NULL;
        }
        subs->val = MakeUnaryCall(conv,copy_expression(val));
    }
    free_basic(bval);
    return subs;
}

static void 
free_substitution(p_substitution subs)
{
  if (subs) {
    free_expression(subs->val);
    free(subs);
  }
}

static bool no_write_effects_on_var(entity var, list le)
{
    FOREACH(EFFECT, e, le)
        if (effect_write_p(e) && entities_may_conflict_p(effect_variable(e), var))
            return false;
    return true;  
}

static bool functionnal_on_effects(reference ref, list /* of effect */ le)
{
    FOREACH(EFFECT, e, le) {
        if ((effect_write_p(e) && effect_variable(e)!=reference_variable(ref)) ||
                (effect_read_p(e) && entities_may_conflict_p(effect_variable(e), reference_variable(ref))))
            return false;
    }
  return true;  
}

/* returns whether there is other write proper effect than on var
 * or if some variable conflicting with var is read... (?) 
 * proper_effects must be available.
 */
static bool functionnal_on(reference ref, statement s)
{
  effects efs = load_proper_rw_effects(s);

  ifdebug(1) {
    pips_assert("efs is consistent", effects_consistent_p(efs));
  }

  return functionnal_on_effects(ref, effects_effects(efs));
}

/* Whether it is a candidate of a substitution operation. that is: 
 * (1) it is a call
 * (2) it is an assignment call
 * (3) the assigned variable is a scalar (sg: no longer true !)
 * (4) there are no side effects in the expression
 * 
 * Note: a substitution candidate might be one after substitutions...
 *       but not before? So effects should be recomputed? or updated?
 * eg: x = 1; x = x + 1;
 */
static p_substitution substitution_candidate(statement s, bool only_scalar)
{
    only_scalar=true;
  list /* of expression */ args;
  call c;
  entity fun;
  syntax svar;
  instruction i = statement_instruction(s);
  
  if (!instruction_call_p(i)) return NULL; /* CALL */
  
  c = instruction_call(i);
  fun = call_function(c);
  
  if (!ENTITY_ASSIGN_P(fun)) return NULL; /* ASSIGN */
  
  
  args = call_arguments(c);
  pips_assert("2 args to =", gen_length(args)==2);
  svar = expression_syntax(EXPRESSION(CAR(args)));

  if(!syntax_reference_p(svar)) // What about subscripts ?
    return NULL;

  reference ref = syntax_reference(svar);
  
  //if (only_scalar && ENDP(reference_indices(ref))) return NULL; /* SCALAR */
  
  if (!functionnal_on(ref, s)) return NULL; /* NO SIDE EFFECTS */
  
  return make_substitution(s, ref, EXPRESSION(CAR(CDR(args))));
}

/* x    = a(i) ; 
 * a(j) = x ;
 *
 * we can substitute x but it cannot be continued.
 * just a hack for this very case at the time. 
 * maybe englobing parallel loops or dependence information could be use for 
 * a better decision? I'll think about it on request only.
 */
static bool cool_enough_for_a_last_substitution(statement s)
{
    p_substitution x = substitution_candidate(s, false);
    bool ok = (x!=NULL);
    free_substitution(x);
    return ok;
}

/* s = r ;
 * s = s + w ; // read THEN write...
 */
static bool other_cool_enough_for_a_last_substitution(statement s, reference ref)
{
  instruction i = statement_instruction(s);
  call c;
  list args;
  syntax svar;
  entity var;
  list le;

  if (!instruction_call_p(i)) 
    return false;

  c = instruction_call(i);
  if (!ENTITY_ASSIGN_P(call_function(c)))
    return false;

  /* it is an assignment */
  args = call_arguments(c);
  pips_assert("2 args to =", gen_length(args)==2);
  svar = expression_syntax(EXPRESSION(CAR(args)));
  pips_assert("assign to a reference", syntax_reference_p(svar));
  var = reference_variable(syntax_reference(svar));
  
  if (!entity_scalar_p(var)) return false;
  
  le = proper_effects_of_expression(EXPRESSION(CAR(CDR(args))));
  bool cool = no_write_effects_on_var(reference_variable(ref), le);
  gen_full_free_list(le);

  return cool;
}

/* do perform the substitution var -> val everywhere in s
 */
static void perform_substitution_in_expression(expression e, p_substitution subs)
{
    syntax s = expression_syntax(e);
    if (syntax_reference_p(s)) {

        reference r = syntax_reference(s);
        if (reference_equal_p(r,subs->ref)) 
        {
            ifdebug(2) {
                pips_debug(2,"substituing %s to %s\n",words_to_string(words_expression(subs->val,NIL)),words_to_string(words_reference(r,NIL)));
            }
            /* FI->FC: the syntax may be freed but not always the reference it
               contains because it can also be used in effects. The bug showed on
               transformations/Validation/fs01.f, fs02.f, fs04.f. I do not know why the
               effects are still used after the statement has been updated (?). The bug
               can be avoided by closing and opening the workspace which generates
               independent references in statements and in effects. Is there a link with
               the notion of cell = reference+preference? */
            expression_syntax(e)=syntax_undefined;
            update_expression_syntax(e,copy_syntax(expression_syntax(subs->val)));
        }
        else
        {
            ifdebug(2) {
                pips_debug(2,"not substituing ");
                print_reference(subs->ref);
                fputs(" to ",stderr);
                print_reference(r);
                fputs("\n",stderr);
            }
        }
    }
}

static bool
call_flt(call c, p_substitution subs)
{
    entity op = call_function(c);
    if(same_entity_p(op, entity_intrinsic(ASSIGN_OPERATOR_NAME))||
            !entity_undefined_p(update_operator_to_regular_operator(op)))
    {
        expression lhs = binary_call_lhs(c);
        if(expression_reference_p(lhs) && reference_equal_p(expression_reference(lhs),subs->ref))
            gen_recurse_stop(lhs);
    }
    return true;
}

static void
perform_substitution(
    p_substitution subs, /* substitution to perform */
    void * s /* where to do this */)
{
    gen_context_multi_recurse(s, subs,
            expression_domain, gen_true, perform_substitution_in_expression,
            call_domain,call_flt,gen_null,
            NULL);
    unnormalize_expression(s);
}

static void
perform_substitution_in_assign(p_substitution subs, statement s)
{
    /* special case */
    pips_assert("assign call", statement_call_p(s));

    call c = statement_call(s);
    perform_substitution(subs,binary_call_rhs(c));
    FOREACH(EXPRESSION, e, reference_indices(expression_reference(binary_call_lhs(c))))
        perform_substitution(subs, e);
}

/* whether there are some conflicts between s1 and s2 
 * according to successor table successors
 */
static bool 
some_conflicts_between(hash_table successors, statement s1, statement s2, p_substitution sub )
{
    pips_debug(2, "looking for conflict with statement\n");
    ifdebug(2) { print_statement(s2); }
    list s1_successors = hash_get(successors,s1);
    FOREACH(SUCCESSOR,s,s1_successors)
    {
        statement ss = vertex_to_statement(successor_vertex(s));
        pips_debug(2, "checking:\n");
        ifdebug(2) { print_statement(ss); }
        if(statement_in_statement_p(ss,s2))
        {
            FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(s)))
            {
                /* if there is a write-* conflict, we cannot do much */
                if ( effect_write_p(conflict_sink(c)) /*&& effect_write_p(conflict_sink(c))*/ )
                {
                    /* this case is safe */
                    if( ENDP(reference_indices(effect_any_reference(conflict_source(c)))) && 
                            !ENDP(reference_indices(effect_any_reference(conflict_sink(c)))))
                        continue;
                    ifdebug(2) {
                        pips_debug(2, "conflict found on reference, with") ;
                        print_reference(effect_any_reference(conflict_sink(c)));
                        print_reference(effect_any_reference(conflict_source(c)));
                        fputs("\n",stderr) ;
                    }
                    return true;
                }
            }
        }
    }

    pips_debug(2, "no conflict\n");
    return false;
}

struct s_p_s {
    p_substitution subs;
    hash_table successors;
    bool stop;
};

static bool
do_substitute(statement anext, struct s_p_s *param)
{
    if(!statement_block_p(anext))
    {
        ifdebug(1) {
            pips_debug(1, "with statement:\n");
            print_statement(anext);
        }
        if (some_conflicts_between(param->successors,param->subs->source, anext, param->subs))
        {
            /* for some special case the substitution is performed.
             * in some cases effects should be updated?
             */
            if (cool_enough_for_a_last_substitution(anext) &&
                    !some_conflicts_between(param->successors,param->subs->source, anext, param->subs))
                perform_substitution(param->subs, anext);
            else
                if (other_cool_enough_for_a_last_substitution(anext, param->subs->ref))
                    perform_substitution_in_assign(param->subs, anext);
            param->stop=true;

            /* now STOP propagation!
            */
            gen_recurse_stop(NULL);
        }
        else
            perform_substitution(param->subs, anext);
    }   
    return true;
}

/* top-down forward substitution of reference in SEQUENCE only.
 */
static bool
fs_filter(statement stat, graph dg)
{
    if(statement_block_p(stat))
    {
        ifdebug(1) {
            pips_debug(1, "considering block statement:\n");
            print_statement(stat);
        }
        hash_table successors = statements_to_successors(statement_block(stat),dg);
        for(list ls = statement_block(stat);!ENDP(ls);POP(ls)) 
        {
            statement first = STATEMENT(CAR(ls));
            if(assignment_statement_p(first))
            {
                ifdebug(1) {
                    pips_debug(1, "considering assignment statement:\n");
                    print_statement(first);
                }
                p_substitution subs = substitution_candidate(first, true);
                if (subs)
                {
                    /* scan following statements and substitute while no conflicts.
                    */
                    struct s_p_s param = { subs, successors,false};
                    bool once = false;
                    FOREACH(STATEMENT,next,CDR(ls))
                    {
                        param.stop=false;
                        gen_context_recurse(next,&param,statement_domain,do_substitute,gen_null);
                        if(param.stop) break;
                        else once=true;
                    }

                    free_substitution(subs);
                    if(once && get_bool_property("FORWARD_SUBSTITUTE_OPTIMISTIC_CLEAN"))
                        update_statement_instruction(first,make_instruction_block(NIL));
                }
            }
        }
        hash_table_free(successors);
    }
    return true;
}

/* interface to pipsmake.
 * should have proper and cumulated effects...
 */
bool forward_substitute(const char* module_name)
{
    debug_on("FORWARD_SUBSTITUTE_DEBUG_LEVEL");

    /* set require resources.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name, true));
    set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
	set_ordering_to_statement(get_current_module_statement());
    graph dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    /* do the job here:
     */
    gen_context_recurse(get_current_module_statement(), dg,statement_domain, fs_filter,gen_null);

    /* return result and clean.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    reset_cumulated_rw_effects();
    reset_proper_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_ordering_to_statement();

    debug_off();

    return true;
}
