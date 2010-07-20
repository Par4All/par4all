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
#include "text-util.h"
#include "pipsdbm.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "sac.h"
#include "ricedg.h"
#include "control.h"
#include "callgraph.h"

#include "effects-convex.h"
#include "atomizer.h"


static hash_table matches = NULL;

/*
 * This function stores in the matches hash table the
 * simd pattern matches for each statement
 */
static void init_statement_matches_map(list l)
{
    matches = hash_table_make(hash_pointer, 0);
    FOREACH(STATEMENT, s,l)
    {
        list match = match_statement(s);

        if (match != NIL)
            hash_put(matches, (void *)s, (void *)match);
    }
}

/*
   This function frees the matches hash table
   */
static void free_statement_matches_map()
{
    hash_table_free(matches);
}

/*
   This function gets the simd pattern matches
   for the statement s
   */
static list get_statement_matches(statement s)
{
    list m =(list)hash_get_default_empty_list(matches,s);
    return  m; 
}

/*
   This function gets the simd pattern matches
   for the statement s checking that the opcodeClass
   of the match is kind
   */
match get_statement_match_of_kind(statement s, opcodeClass kind)
{
    list l = get_statement_matches(s);

    for( ; l!=NIL; l=CDR(l))
    {
        match m = MATCH(CAR(l));
        if (match_type(m) == kind)
            return m;
    }

    return match_undefined;
}

/*
 * This function gets the simd pattern matches opcodeClass's
 * list for the statement s
 */
static set get_statement_matching_types(statement s)
{
    set out = set_make(set_pointer);

    FOREACH(MATCH,m,get_statement_matches(s))
        set_add_element(out,out,match_type(m));

    return out;
}

static hash_table equivalence_table = hash_table_undefined;

static bool successor_only_has_rr_conflict_p(successor su)
{
    bool all_rr = true;
    FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(su)))
    {
        if(effect_read_p(conflict_sink(c))&&
                effect_read_p(conflict_source(c))
          )
        {
            pips_debug(3,
                    "conflict skipped between %s and %s\n",
                    words_to_string(words_reference(effect_any_reference(conflict_sink(c)),NIL)),
                    words_to_string(words_reference(effect_any_reference(conflict_source(c)),NIL)));
        }
        else {
            all_rr = false;
            pips_debug(3,
                    "conflict found between %s and %s\n",
                    words_to_string(words_reference(effect_any_reference(conflict_sink(c)),NIL)),
                    words_to_string(words_reference(effect_any_reference(conflict_source(c)),NIL)));
        }
    }
    return all_rr;
}

/*
 * This function stores in the hash_table equivalence_table
 * all statement equivalent to those in l
 * 
 * it is done by iterating over the `dependence_graph'
 */
static void init_statement_equivalence_table(list l,graph dependence_graph)
{
    pips_assert("free was called",hash_table_undefined_p(equivalence_table));
    equivalence_table=hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);

    /* for faster access */
    set statements = set_make(set_pointer);
    set_assign_list(statements,l);
    hash_table counters = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);

    /* first extract corresponding vertices */
    FOREACH(VERTEX, a_vertex,graph_vertices(dependence_graph))
    {
        statement s = vertex_to_statement(a_vertex);
        if(set_belong_p(statements,s))
            hash_put(counters,a_vertex,(void*)0);
    }
    /* then count the references between each other */
    for(void *k,*v,*iter=NULL; (iter=hash_table_scan(counters,iter,&k,&v));)
    {
        FOREACH(SUCCESSOR,su,vertex_successors((vertex)k))
        {
            /* do not take into account backward references, or R-R conflicts */
            if(vertex_ordering(successor_vertex(su)) > vertex_ordering((vertex)k)  &&
                    !successor_only_has_rr_conflict_p(su) )
            {
                void* counter = hash_get(counters,successor_vertex(su));
                if(counter != HASH_UNDEFINED_VALUE)
                {
                    intptr_t value = (intptr_t)counter;
                    ++value;
                    pips_debug(4,"counter now is %td for\n",value);
                    ifdebug(4) print_statement(vertex_to_statement(successor_vertex(su)));
                    pips_debug(4,"referenced by\n");
                    ifdebug(4) print_statement(vertex_to_statement((vertex)k));
                    hash_update(counters,successor_vertex(su),(void*)value);
                }
            }
        }
    }

    /* now  recursievly retreive the head of each vertex with no reference on them */
    while(!hash_table_empty_p(counters))
    {
        set head = set_make(set_pointer);
        /* fill the head */
        HASH_MAP(k,v,if((intptr_t)v == 0) set_add_element(head,head,k), counters);
        /* found nothing, assume we are in the tail */
        if(set_empty_p(head))
        {
            list equivalence_list = NIL;
            HASH_MAP(k,v,equivalence_list=CONS(STATEMENT,vertex_to_statement((vertex)k),equivalence_list),counters);
            HASH_MAP(k,v, hash_put(equivalence_table,vertex_to_statement((vertex)k),equivalence_list), counters);
            hash_table_clear(counters);
        }
        /* remove those vertex from the head and decrease references elsewhere */
        else
        {
            list equivalence_list = NIL;
            { SET_FOREACH(vertex,v,head) equivalence_list=CONS(STATEMENT,vertex_to_statement(v),equivalence_list); }

            SET_FOREACH(vertex,v,head) {
                FOREACH(SUCCESSOR,su,vertex_successors(v)) {
                    void* counter = hash_get(counters,successor_vertex(su));
                    /* do not take into account backward references and ignored statements */
                    if(counter != HASH_UNDEFINED_VALUE && vertex_ordering(successor_vertex(su)) > vertex_ordering(v) && !successor_only_has_rr_conflict_p(su))
                    {
                        intptr_t value = (intptr_t)counter;
                        --value;
                        pips_assert("not missed something",value>=0);
                        hash_update(counters,successor_vertex(su),(void*)value);
                    }
                }
                hash_put(equivalence_table,vertex_to_statement(v),equivalence_list);
                hash_del(counters,v);
            }
        }
        set_free(head);
    }
    hash_table_free(counters);
}

/*
 * This function frees the successors list
 */
static void free_statement_equivalence_table()
{
    pips_assert("init called",!hash_table_undefined_p(equivalence_table));
    hash_table_free(equivalence_table);/* leak spotted !*/
    equivalence_table=hash_table_undefined;
}

static set extract_non_conflicting_statements(statement s, list tail)
{
    list equivalences = (list)hash_get(equivalence_table,s);
    pips_assert("table is correct",equivalences!= HASH_UNDEFINED_VALUE);

    set eq = set_make(set_pointer);
    set_assign_list(eq,equivalences);

    set stail = set_make(set_pointer);
    set_assign_list(stail,tail);

    set_intersection(eq,eq,stail);
    set_free(stail);
    ifdebug(3) {
        pips_debug(3,"non conficting statement with\n");
        print_statement(s);
        SET_FOREACH(statement,st,eq)
            print_statement(st);
    }

    return eq;
}
static set extract_matching_statements(statement s, set stats,set *matches)
{
    set out = set_make(set_pointer);
    SET_FOREACH(statement,st,stats)
    {
        set smt = get_statement_matching_types(st);
        set tmp = set_make(set_pointer);
        set_intersection(tmp,*matches,smt);
        if(set_empty_p(tmp))
        {
            set_free(tmp);
        }
        else {
            simd_fill_curArgType(st);
            if( simd_check_argType()){
                *matches=tmp;
                set_add_element(out,out,st);
            }
            else {
                ifdebug(3){
                    pips_debug(3,"following statement does not have good arg type\n");
                    print_statement(st);
                }
            }
        }
        set_free(smt);
    }
    return out;
}

static bool sac_statement_to_expressions_gather(expression e,list *l)
{
    if(expression_reference_or_field_p(e))
    {
        *l=CONS(EXPRESSION,e,*l);
        return false;
    }
    return true;
}

static list sac_statement_to_expressions(statement s)
{
    list out = NIL;
    gen_context_recurse(s,&out,expression_domain,sac_statement_to_expressions_gather,gen_null);
    return out;
}


static void sac_sort_expressions_reductions_last(list *l)
{
    list tail = NIL;
    list head = NIL;
    FOREACH(EXPRESSION,exp,*l)
        if(sac_expression_reduction_p(exp))
            tail=CONS(EXPRESSION,exp,tail);
        else
            head=CONS(EXPRESSION,exp,head);
    tail=gen_nreverse(tail);
    head=gen_nreverse(head);
    gen_free_list(*l);
    *l=gen_nconc(head,tail);
}

static bool comparable_statements_on_distance_p(statement s0, statement s1)
{
    list exp0=sac_statement_to_expressions(s0),
        exp1=sac_statement_to_expressions(s1);
    list iter=exp1;
    FOREACH(EXPRESSION,e0,exp0)
    {
        expression e1 = EXPRESSION(CAR(iter));
        expression distance = distance_between_expression(e1,e0);
        intptr_t val=0;
        if(!expression_undefined_p(distance))
        {
            (void)expression_integer_value(distance,&val);
            free_expression(distance);
            if(val) return true;
        }
        POP(iter);
        if(ENDP(iter)) break;
    }
    return false;
}
static int compare_statements_on_ordering(const void * v0, const void * v1)
{
    const statement s0 = *(const statement*)v0;
    const statement s1 = *(const statement*)v1;
    if (statement_ordering(s0) > statement_ordering(s1)) return 1;
    if (statement_ordering(s0) < statement_ordering(s1)) return -1;
    return 0;
}

static int compare_statements_on_distance(const void * v0, const void * v1)
{
    const statement s0 = *(const statement*)v0;
    const statement s1 = *(const statement*)v1;
    list exp0=sac_statement_to_expressions(s0),
        exp1=sac_statement_to_expressions(s1);
    /* hook for reductions :
     * it is less painful to load reduction in a bad order than other reference,
     * so we reorder the expressions exp0 in order to check reductions last
     */
    sac_sort_expressions_reductions_last(&exp0);
    sac_sort_expressions_reductions_last(&exp1);
    list iter=exp1;
    int res = 0;
    FOREACH(EXPRESSION,e0,exp0)
    {
        expression e1 = EXPRESSION(CAR(iter));
        expression distance = distance_between_expression(e1,e0);
        intptr_t val=0;
        if(!expression_undefined_p(distance))
        {
            (void)expression_integer_value(distance,&val);
            free_expression(distance);
            if((res=val)) break;
        }
        POP(iter);
    }
    gen_free_list(exp0);
    gen_free_list(exp1);
    return res?res:compare_statements_on_ordering(v0,v1);
}
static int compare_list_from_length(const void *v0, const void *v1)
{
    const list l0=*(const list*)v0;
    const list l1=*(const list*)v1;
    size_t n0 = gen_length(l0);
    size_t n1 = gen_length(l1);
    return n0==n1 ? 0 : n0 > n1 ? 1 : -1;
}

static list order_isomorphic_statements(set s)
{
    list ordered = set_to_sorted_list(s,compare_statements_on_ordering);
    list heads = NIL;
    do {
        statement head = STATEMENT(CAR(ordered));
        list firsts = CONS(STATEMENT,head,NIL);
        list lasts = NIL;
        FOREACH(STATEMENT,st,CDR(ordered))
        {
            if(comparable_statements_on_distance_p(head,st))
                firsts=CONS(STATEMENT,st,firsts);
            else
                lasts=CONS(STATEMENT,st,lasts);
        }
        gen_free_list(ordered);
        gen_sort_list(firsts,compare_statements_on_distance);
        gen_sort_list(lasts,compare_statements_on_ordering);
        heads=CONS(LIST,firsts,heads);
        ordered=lasts;
    } while(!ENDP(ordered));
    gen_sort_list(heads,compare_list_from_length);
    list out = NIL;
    FOREACH(LIST,l,heads)
        out=gen_nconc(l,out);
    gen_free_list(heads);
    return out;
}

static list simdize_simple_statements_pass2(list seq, float * simdCost)
{
    list newseq=NIL;

    //argument info dependencies are local to each sequence -> RESET
    init_vector_to_expressions();

    set visited = set_make(set_pointer);

    /* Traverse to list to group isomorphic statements */
    for(list  i = seq;!ENDP(i);POP(i))
    {
        statement si = STATEMENT(CAR(i));
        if(!set_belong_p(visited,si))
        {
            set_add_element(visited,visited,si);
            ifdebug(3){
                pips_debug(3,"trying to match statement\n");
                print_statement(si);
            }

            /* if this is not a recognized statement (ie, no match), skip it */
            set group_matches = get_statement_matching_types(si);
            if (set_empty_p(group_matches) )
            {
                pips_debug(3,"no match found\n");
                invalidate_expressions_in_statement(si);
                newseq = CONS(STATEMENT,si,newseq);
                continue;
            }
            else 
            {
                ifdebug(3) {
                    pips_debug(3,"matching opcode found:\n");
                    SET_FOREACH(opcodeClass,oc,group_matches)
                        pips_debug(3,"%s\n",opcodeClass_name(oc));
                }
            }

            simd_fill_finalArgType(si);
            ifdebug(3) {
                pips_debug(3,"examining following statments:\n");
                print_statements(CDR(i));
            }

            /* try to find all the compatible isomorphic statements after the
             * current statement
             */
            /* if the matches for statement sj and for the group have a non-empty
             * intersection, and the move is legal (ie, does not break dependency
             * chain) then we can add the statement sj to the group.
             */
            set non_conflicting = extract_non_conflicting_statements(si,CDR(i));
            /* filter out already visited statements */
            set_difference(non_conflicting,non_conflicting,visited);
            set isomorphics = extract_matching_statements(si,non_conflicting,&group_matches);
            if(set_empty_p(isomorphics))
            {
                ifdebug(3){
                    pips_debug(3,"following statement cannot be moved\n");
                    print_statement(si);
                }
                invalidate_expressions_in_statement(si);
                newseq = CONS(STATEMENT,si,newseq);
            }
            else
            {
                set_add_element(isomorphics,isomorphics,si);
                ifdebug(3){
                    pips_debug(3,"following statements matches!\n");
                    SET_FOREACH(statement,sj,isomorphics)
                        print_statement(sj);
                }
                list iso_stats = order_isomorphic_statements(isomorphics);
                set_free(isomorphics);
                for(list iso_iter=iso_stats;!ENDP(iso_iter);)
                {
                    simdstatement ss = make_simd_statements(group_matches, iso_iter);
                    if(simdstatement_undefined_p(ss))
                    {
                        invalidate_expressions_in_statement(si);
                        newseq = CONS(STATEMENT,si,newseq);
                        POP(iso_iter);
                        break;
                    }
                    else
                    {
                        newseq=gen_append(generate_simd_code(ss,simdCost),newseq);
                        for(intptr_t i= opcode_vectorSize(simdstatement_opcode(ss));i!=0;i--)
                        {
                            if(!ENDP(iso_iter)) { /*endp can occur when padded statements are added */
                                set_add_element(visited,visited,STATEMENT(CAR(iso_iter)));
                                POP(iso_iter);
                            }
                        }
                    }
                }
                set_free(group_matches);
                gen_free_list(iso_stats);
            }
            simd_reset_finalArgType();
        }
    }
    reset_vector_to_expressions();
    newseq=gen_nreverse(newseq);

    /* Free the list of statements info */

    /* Set the new list as the statements' instructions */
    return newseq;
}

statement sac_current_block = statement_undefined;
instruction sac_real_current_instruction = instruction_undefined;

/*
 * This function tries to simdize with two algorithms.
 *
 * `simdize_simple_statements_pass1' attempts to simdize by grouping
 * the statements that have the same simd number.
 *
 * `simdize_simple_statements_pass2' attempts to simdize by grouping
 * as many statements as possible together.
 */
static void simdize_simple_statements(statement s,graph dependence_graph)
{
    /* not much we can do with a single statement, or with
     * "complex" statements (ie, with tests/loops/...)
     */
    if (statement_block_p(s))
    {
        sac_current_block=s;
        /* we cannot handle anything but sequence of calls */
        list iter = statement_block(s);
        sac_real_current_instruction = statement_instruction(s);
        statement_instruction(s)=make_instruction_block(NIL);

        list new_seq = NIL;
        while(!ENDP(iter))
        {
            list seq = NIL;
            for(;!ENDP(iter);POP(iter))
            {
                statement st = STATEMENT(CAR(iter));
                if(declaration_statement_p(st))
                    insert_statement(s,st,false);
                else {
                    if(statement_call_p(st))
                        seq=CONS(STATEMENT,st,seq);

                    if(!statement_call_p(st)||ENDP(CDR(iter)))
                    {
                        /* process already existing statements */
                        if(!ENDP(seq))
                        {
                            seq=gen_nreverse(seq);

                            init_statement_matches_map(seq);
                            init_statement_equivalence_table(seq,dependence_graph);

                            float saveSimdCost = 0;
                            list simdseq = simdize_simple_statements_pass2(seq, &saveSimdCost);


                            pips_debug(2,"opcode cost1 %f\n", saveSimdCost);

                            if((saveSimdCost >= 0.0001))
                            {
                                new_seq=gen_append(new_seq,seq);
                                gen_free_list(simdseq);
                            }
                            else
                            {
                                new_seq=gen_append(new_seq,simdseq);
                                gen_free_list(seq);
                            }

                            free_statement_matches_map();
                            free_statement_equivalence_table();
                            seq=NIL;

                        }
                        if(!statement_call_p(st))
                            new_seq=gen_append(new_seq,CONS(STATEMENT,st,NIL));
                    }
                }

            }
        }
        sequence_statements(instruction_sequence(statement_instruction(s)))=
            gen_append(statement_block(s),new_seq);
    }
}

/*
 * simple filter for `simdizer'
 * Do not recurse through simple calls, for better performance 
 */
static bool simd_simple_sequence_filter(statement s,__attribute__((unused)) graph dg )
{
    return ! ( instruction_call_p( statement_instruction(s) ) ) ;
}

string sac_commenter(entity e)
{
    if(sac_aligned_entity_p(e))
    {
        return strdup("SAC generated temporary array");
    }
    else if(simd_vector_entity_p(e))
    {
        string s,g = basic_to_string(entity_basic(e));
        asprintf(&s,"PIPS:SAC generated %s vector(s)", g);
        free(g);
        return s;
    }
    else
        return strdup("PIPS:SAC generated variable");
}


/*
 * main entry function
 * basically run `simdize_simple_statements' on all sequences
 */
bool simdizer(char * mod_name)
{
    /* get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, TRUE);

    set_current_module_statement(mod_stmt); 
    set_current_module_entity(module_name_to_entity(mod_name));
	set_ordering_to_statement(mod_stmt);
    graph dependence_graph = 
        (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);
    set_simd_treematch((matchTree)db_get_memory_resource(DBR_SIMD_TREEMATCH,"",TRUE));
    set_simd_operator_mappings(db_get_memory_resource(DBR_SIMD_OPERATOR_MAPPINGS,"",TRUE));
    set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,mod_name,TRUE));
    push_generated_variable_commenter(sac_commenter);
    init_padding_entities();

    debug_on("SIMDIZER_DEBUG_LEVEL");
    /* Now do the job */

    gen_context_recurse(mod_stmt, dependence_graph, statement_domain,
            simd_simple_sequence_filter, simdize_simple_statements);

    pips_assert("Statement is consistent after SIMDIZER", 
            statement_consistent_p(mod_stmt));

    /* Reorder the module, because new statements have been added */  
    clean_up_sequences(mod_stmt);
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, compute_callees(mod_stmt));
    //DB_PUT_MEMORY_RESOURCE(DBR_DG, mod_name, dependence_graph);

    /* update/release resources */
    reset_padding_entities();
    reset_cumulated_rw_effects();
    reset_simd_operator_mappings();
    reset_simd_treematch();
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    pop_generated_variable_commenter();

    debug_off();

    return TRUE;
}


static void do_simdizer_init(call c)
{
#define SWAP_ARGUMENTS(c) do { call_arguments(c)=gen_nreverse(call_arguments(c)) ; return ;} while(0)
#define NOSWAP_ARGUMENTS(c)  return
    if(commutative_call_p(c))
    {
        pips_assert("commutative call are binary calls",gen_length(call_arguments(c))==2);
        expression e0 = binary_call_lhs(c),
                   e1 = binary_call_rhs(c);
        /* constant first */
        if(extended_expression_constant_p(e0)) NOSWAP_ARGUMENTS(c);
        if(extended_expression_constant_p(e1)) SWAP_ARGUMENTS(c);
        /* then scalar */
        if(expression_scalar_p(e0)) NOSWAP_ARGUMENTS(c);
        if(expression_scalar_p(e1)) SWAP_ARGUMENTS(c) ;

        /* we finally end with two references */
        if(expression_reference_or_field_p(e0) && expression_reference_or_field_p(e1))
        {
            expression distance = distance_between_expression(e0,e1);
            intptr_t val;
            if( !expression_undefined_p(distance) && expression_integer_value(distance,&val))
            {
                free_expression(distance);
                if(val <= 0) NOSWAP_ARGUMENTS(c);
                else SWAP_ARGUMENTS(c) ;
            }
            else
            {
                entity ent0 = expression_to_entity(e0);
                entity ent1 = expression_to_entity(e1);
                if(compare_entities(&ent0,&ent1)<=0) NOSWAP_ARGUMENTS(c);
                else SWAP_ARGUMENTS(c) ;
            }
        }
        else
        {
            string msg = words_to_string(words_call(c,0,true,true,NIL));
            pips_user_warning("unable to decide how to commute %s\n",msg);
            free(msg);
        }
    }
#undef SWAP_ARGUMENTS
#undef NOSWAP_ARGUMENTS
}
bool simdizer_init(const char * module_name)
{
    /* get the resources */
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name, true));
    set_current_module_entity(module_name_to_entity(module_name));

    /* sort commutative operators */
    gen_recurse(get_current_module_statement(),call_domain,gen_true,do_simdizer_init);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /* reset */
    reset_current_module_statement();
    reset_current_module_entity();

    return true;

}
