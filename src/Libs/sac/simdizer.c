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

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "transformations.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"
#include "control.h"
#include "callgraph.h"

#include "effects-convex.h"
#include "sac.h"
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
        set succ = set_make(set_pointer);
        statement s = vertex_to_statement(a_vertex);
        if(set_belong_p(statements,s))
            hash_put(counters,a_vertex,(void*)0);
    }
    /* then count the references between each other */
    for(void *k,*v,*iter=NULL; (iter=hash_table_scan(counters,iter,&k,&v));)
    {
        FOREACH(SUCCESSOR,su,vertex_successors((vertex)k))
        {
            /* do not take into account backward references */
            if(vertex_ordering(successor_vertex(su)) > vertex_ordering((vertex)k)  )
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
                    if(counter != HASH_UNDEFINED_VALUE && vertex_ordering(successor_vertex(su)) > vertex_ordering(v) )
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

#if 0
/*
   This function returns TRUE if there is a conflict between s1 and s2
   that prevents the simdization of s2
   */
static bool successor_p(statement s1, statement s2, bool nonGroupStat)
{
    list succ;

    if(!instruction_call_p(statement_instruction(s1)))
    {
        return TRUE;
    }

    succ = (list)hash_get(successors, (void*)s1);

    FOREACH(SUCCESSOR, s,succ)
    {
        if (vertex_to_statement(successor_vertex(s)) == s2)
        {
            FOREACH(CONFLICT, c,dg_arc_label_conflicts(successor_arc_label(s)))
            {
                // If there is a write-read conflict between
                // s1-s2, then s2 can't be simdized
                if((effect_write_p(conflict_source(c)) && 
                            effect_read_p(conflict_sink(c))))
                {
                    ifdebug(4) {
                        pips_debug(4,"write read conflict between:\n");
                        print_effect(conflict_source(c));
                        print_effect(conflict_sink(c));
                    }
                    return TRUE;
                }

                // If there is a read-write conflict or 
                // a write-write conflict between s1-s2
                // then s2 can't be simdized. Because, otherwise,
                // since s1 doesn't belong to the same simd
                // group as s2, then s2 would be executed before s1.
                if(((effect_read_p(conflict_source(c)) && 
                                effect_write_p(conflict_sink(c)))) ||
                        ((effect_write_p(conflict_source(c)) && 
                          effect_write_p(conflict_sink(c)))))
                {
                    if(nonGroupStat)
                    {
                        ifdebug(4) {
                            pips_debug(4,"read write conflict between:\n");
                            print_effect(conflict_source(c));
                            print_effect(conflict_sink(c));
                        }
                        return TRUE;
                    }
                }

            }
        }
    }
    return FALSE;
}
#endif

#define SIMD_COMMENT "SIMD_COMMENT_"

/*
   This function return TRUE if SIMD_COMMENT_ has been
   found in the statement comments. And, if so, num holds
   the simd number of the statement.
   */
static bool getSimdCommentNum(statement stat, int * num) 
{
    string comments;
    char*  next_line;

    bool res = FALSE;

    if (!statement_with_empty_comment_p(stat)) 
    {
        comments = strdup(statement_comments(stat));
        next_line = strtok (comments, "\n");
        if (next_line != NULL) {
            do {
                sscanf(next_line, "c SIMD_COMMENT_%d", num);
                res = TRUE;

                next_line = strtok(NULL, "\n");
            }
            while (next_line != NULL);
        }
        free(comments);
    }

    return res;
}
#if 0

/*
   This function returns TRUE if s can be added to the simd group
   whose STATEMENT(CAR(group_first)) is the first element
   */
static bool move_allowed_p(list group_first, list group_last, statement s)
{ 
    cons * i;
    bool nonGroupStat = FALSE;

    for(i = group_first; (i != NIL) && (STATEMENT(CAR(i)) != s); i = CDR(i))
    {
        if (successor_p(STATEMENT(CAR(i)), s, nonGroupStat))
        {
            pips_debug(3,"move not allowed\n");
            return FALSE;
        }

        if((STATEMENT(CAR(group_last)) == STATEMENT(CAR(i))) && (!nonGroupStat))
        {
            nonGroupStat = TRUE;
        }
    }
    pips_debug(3,"move allowed\n");

    return TRUE;
}
#endif
#if 0

/* Transform the code to use SIMD instructions. The input statement
 * should be a sequence of simple statements.
 * If the block contains loops, ifs, or any other control flow 
 * instruction, the result may not be correct.
 * Non-simple calls should be handled properly, but WILL NOT be 
 * optimized to use SIMD instructions.
 */
static list simdize_simple_statements_pass1(list seq, float * simdCost)
{
    cons * i;
    list sinfo; /* <statement_info> */
    list sinfo_begin;
    list newseq;

    //argument info dependencies are local to each sequence -> RESET
    reset_argument_info();

    // cons a NULL?
    sinfo = sinfo_begin = gen_statementInfo_cons( make_statementInfo(0,NULL), NIL);

    /* Traverse to list to group isomorphic statements */
    for( i = seq;
            i != NIL;
            i = CDR(i) )
    {
        cons * j, * p;
        cons * group_first, * group_last;
        statement si = STATEMENT(CAR(i));

        /* Initialize current group */
        group_first = i;
        group_last = i;

        set group_matches = get_statement_matching_types(si);

        //printf("si\n");print_statement(si);
        /* try to find all the compatible isomorphic statements after the
         * current statement
         */
        for( j = CDR(group_last), p = NIL;
                j != NIL;
                p = j, j = CDR(j) )
        {
            statement sj = STATEMENT(CAR(j));
            int num1, num2;
            list m_sj;

            //printf("sj\n");print_statement(sj);
            /* if the two statements came from the same original statement
            */
            if(getSimdCommentNum(si, &num1) &&
                    getSimdCommentNum(sj, &num2))
            {
                if(num1 != num2)
                {
                    continue;
                }
            }
            else
            {
                continue;
            }

            m_sj = get_statement_matching_types(sj);

            gen_list_and(&group_matches, m_sj);
            if (move_allowed_p(group_first, group_last, sj))
            {
                if (j != CDR(group_last))
                {
                    /* do the move */
                    CDR(p) = CDR(j);
                    CDR(group_last) = CONS(STATEMENT, sj, CDR(group_last));

                    /* keep searching from the next one */
                    j = p;
                }

                group_last = CDR(group_last);
            }
        }

        /* if this is not a recognized statement (ie, no match), skip it */
        if (group_matches == NIL)
        {
            list ind = group_first;
            /* No optimized opcode found... */
            for( ;
                    (ind != CDR(group_last));
                    ind = CDR(ind) )
            {
                CDR(sinfo) = gen_statementInfo_cons(
                        make_nonsimd_statement_info(STATEMENT(CAR(ind))),
                        NIL);
                sinfo = CDR(sinfo);
            }
        }
        else
        {
            /* the previous group of isomorphic statements is complete. 
             * we can now generate SIMD statement info.
             * group is delimited by group_first and group_last (included)
             */
            CDR(sinfo) = make_simd_statements(group_matches, 
                    group_first, 
                    group_last);
        }

        while(CDR(sinfo) != NIL)
            sinfo = CDR(sinfo);

        /* skip what has already been matched */
        i = group_last;
    }

    /* Now, based on the statement information gathered, 
     * generate the actual code (new sequence of statements)
     */
    newseq = generate_simd_code(CDR(sinfo_begin), simdCost);

    /* Free the list of statements info */
    gen_free_list(sinfo_begin);

    /* Set the new list as the statements' instructions */
    return newseq;
}
/*
   This function returns true if ts1 is a sucessor of s2
   */
static bool dg_successor_p(statement s1, statement s2)
{
    set succ = (set)hash_get(successors, (void*)s2);

    SET_FOREACH(successor, s,succ)
    {
        statement curr = vertex_to_statement(successor_vertex(s));
        if ( curr == s1)
        {
            FOREACH(CONFLICT, c, dg_arc_label_conflicts(successor_arc_label(s)))
            {
                // If there is a write-read conflict between
                // s1-s2, then s2 can't be simdized
                if((effect_write_p(conflict_source(c)) && 
                            effect_read_p(conflict_sink(c))))
                {
                    ifdebug(4) {
                        pips_debug(4,"write read conflict between:\n");
                        print_effect(conflict_source(c));
                        print_effect(conflict_sink(c));
                    }
                    return true;
                }

                // If there is a read-write conflict or 
                // a write-write conflict between s1-s2
                // then s2 can't be simdized. Because, otherwise,
                // since s1 doesn't belong to the same simd
                // group as s2, then s2 would be executed before s1.
                if(((effect_read_p(conflict_source(c)) && 
                                effect_write_p(conflict_sink(c)))) ||
                        ((effect_write_p(conflict_source(c)) && 
                          effect_write_p(conflict_sink(c)))))
                {
                    ifdebug(4) {
                        pips_debug(4,"read write conflict between:\n");
                        print_effect(conflict_source(c));
                        print_effect(conflict_sink(c));
                    }
                    return true;
                }
                /* a write-write conflict is not that good either */
                if((effect_write_p(conflict_source(c)) && 
                            effect_write_p(conflict_sink(c))))
                {
                    ifdebug(4) {
                        pips_debug(4,"write write conflict between:\n");
                        print_effect(conflict_source(c));
                        print_effect(conflict_sink(c));
                    }
                    return true;
                }
            }
        }
    } 
    return false;
}
#endif

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

static bool comparable_statements_on_distance_p(statement s0, statement s1)
{
    list exp0=sac_statement_to_expressions(s0),
        exp1=sac_statement_to_expressions(s1);
    list iter=exp1;
    FOREACH(EXPRESSION,e0,exp0)
    {
        expression e1 = EXPRESSION(CAR(iter));
        expression distance = distance_between_expression(e1,e0);
        int val=0;
        if(!expression_undefined_p(distance))
        {
            (void)expression_integer_value(distance,&val);
            free_expression(distance);
            if(val) return true;
        }
        POP(iter);
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
    list iter=exp1;
    FOREACH(EXPRESSION,e0,exp0)
    {
        expression e1 = EXPRESSION(CAR(iter));
        expression distance = distance_between_expression(e1,e0);
        int val=0;
        if(!expression_undefined_p(distance))
        {
            (void)expression_integer_value(distance,&val);
            free_expression(distance);
            if(val) return val;
        }
        POP(iter);
    }
    return compare_statements_on_ordering(v0,v1);
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
    list newseq;

    //argument info dependencies are local to each sequence -> RESET
    reset_argument_info();

    list sinfo = NIL;
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

            /* Initialize current group */
            list group_current = CONS(STATEMENT,si,NIL);

            /* if this is not a recognized statement (ie, no match), skip it */
            set group_matches = get_statement_matching_types(si);
            if (set_empty_p(group_matches) )
            {
                pips_debug(3,"no match found\n");
                sinfo = gen_statementInfo_cons( make_nonsimd_statement_info(STATEMENT(CAR(i))),sinfo);
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
                sinfo = gen_statementInfo_cons( make_nonsimd_statement_info(si),sinfo);
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
                statementInfo ssi = make_simd_statements(group_matches, iso_stats);
                if(statementInfo_undefined_p(ssi))
                {
                    sinfo = gen_statementInfo_cons( make_nonsimd_statement_info(si),sinfo);
                }
                else
                {
                    sinfo=gen_statementInfo_cons(ssi,sinfo);
                    list iter = iso_stats;
                    for(intptr_t i= opcode_vectorSize(simdStatementInfo_opcode(statementInfo_simd(ssi)));i!=0;i--)
                    {
                        if(!ENDP(iter)) { /*endp can occur when padded statements are added */
                            set_add_element(visited,visited,STATEMENT(CAR(iter)));
                            POP(iter);
                        }
                    }
                }
                gen_free_list(iso_stats);
            }
            simd_reset_finalArgType();
        }
    }
    sinfo=gen_nreverse(sinfo);

    /* Now, based on the statement information gathered, 
     * generate the actual code (new sequence of statements)
     */
    newseq = generate_simd_code(sinfo, simdCost);

    /* Free the list of statements info */
    free_simd_statements(sinfo);
    gen_free_list(sinfo);

    /* Set the new list as the statements' instructions */
    return newseq;
}

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
    list seq = NIL;
    list copyseq = NIL;

    list newseq = NIL;
    list saveseq = NIL;

    float newSimdCost = 0;
    float saveSimdCost = 0;

    if (!instruction_sequence_p(statement_instruction(s)))
        /* not much we can do with a single statement, or with
         * "complex" statements (ie, with tests/loops/...)
         */
        return;

    seq = sequence_statements(instruction_sequence(statement_instruction(s)));

    init_statement_matches_map(seq);
    init_statement_equivalence_table(seq,dependence_graph);


    saveseq = simdize_simple_statements_pass2(seq, &saveSimdCost);


    pips_debug(2,"opcode cost1 %f\n", saveSimdCost);

    if((saveSimdCost >= 0.0001))
    {
        gen_free_list(newseq);
        gen_free_list(saveseq);
    }
    else
    {
        sequence_statements(instruction_sequence(statement_instruction(s))) = saveseq;
        gen_free_list(seq);
    }

    free_statement_matches_map();
    free_statement_equivalence_table();
}

/*
 * simple filter for `simdizer'
 * Do not recurse through simple calls, for better performance 
 */
static bool simd_simple_sequence_filter(statement s,__attribute__((unused)) graph dg )
{
    return ! ( instruction_call_p( statement_instruction(s) ) ) ;
}

static string sac_commenter(entity e)
{
    if(simd_vector_entity_p(e))
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

    //what a trick ! this is needed because we would lose declarations otherwise */
    statement parent_stmt=make_block_statement(make_statement_list(mod_stmt));

    set_current_module_statement(parent_stmt); 
    set_current_module_entity(module_name_to_entity(mod_name));
	set_ordering_to_statement(mod_stmt);
    graph dependence_graph = 
        (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);
    set_simd_treematch((matchTree)db_get_memory_resource(DBR_SIMD_TREEMATCH,"",TRUE));
    set_simd_operator_mappings(db_get_memory_resource(DBR_SIMD_OPERATOR_MAPPINGS,"",TRUE));
    push_generated_variable_commenter(sac_commenter);
    init_padding_entities();

    debug_on("SIMDIZER_DEBUG_LEVEL");
    /* Now do the job */

    gen_context_recurse(mod_stmt, dependence_graph, statement_domain,
            simd_simple_sequence_filter, simdize_simple_statements);

    pips_assert("Statement is consistent after SIMDIZER", 
            statement_consistent_p(mod_stmt));

    /* Reorder the module, because new statements have been added */  
    clean_up_sequences(parent_stmt);
    module_reorder(parent_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, parent_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, compute_callees(parent_stmt));
    //DB_PUT_MEMORY_RESOURCE(DBR_DG, mod_name, dependence_graph);

    /* update/release resources */
    reset_padding_entities();
    reset_simd_operator_mappings();
    reset_simd_treematch();
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    pop_generated_variable_commenter();

    debug_off();

    return TRUE;
}

#if 0
static bool successor_internal_p(statement s1, statement s2,set seen)
{
    list succ = (list)hash_get(successors, (void*)s2);

    FOREACH(SUCCESSOR, s,succ)
    {
        statement curr = vertex_to_statement(successor_vertex(s));
        if(!set_belong_p(seen,curr))
        {
            set_add_element(seen,seen,curr);
            if ( curr == s1)
                return true;
            if( successor_internal_p(s1,curr,seen))
                return true;
        }
    } 
    return false;
}
/*
   This function returns true if ts1 is a sucessor of s2
   */
static bool successor_p(statement s1, statement s2)
{
    set seen = set_make(set_pointer);
    bool res = successor_internal_p(s1,s2,seen);
    set_free(seen);
    return res;
}

static int compare_statements(const void * v0, const void * v1)
{
    statement s0 = *(statement*)v0;
    statement s1 = *(statement*)v1;
    if (statement_ordering(s0) > statement_ordering(s1)) return 1;
    if (statement_ordering(s0) < statement_ordering(s1)) return -1;
    return 0;
}
static list do_statement_packing(list statements)
{
    if(!ENDP(statements))
    {
        list first_pack = NIL;
        list second_pack = NIL;
        statement head = STATEMENT(CAR(statements));
        /* first pack is filled with head successors
         * second pack is filled with others */
        FOREACH(STATEMENT,st,CDR(statements))
            if(dg_successor_p(st,head))
                first_pack=CONS(STATEMENT,st,first_pack);
            else
                second_pack=CONS(STATEMENT,st,second_pack);
        /* repeat the process on second_pack */
        list fst = do_statement_packing(first_pack);
        list snd=NIL;
        if(ENDP(second_pack)) snd=CONS(STATEMENT,head,NIL);
        else {
            do_statement_packing(second_pack);
        }

    }
    return NIL;
}
static list statement_packing(list statements)
{
    list decls = NIL;
    list iter = statements;
    /* move out declarations */
    for(;!ENDP(iter);POP(iter))
    {
        statement st = STATEMENT(CAR(iter));
        if(declaration_statement_p(st))
            decls=CONS(STATEMENT,st,decls);
        else
            break;
    }
    decls=gen_nreverse(decls);
    return CONS(LIST,decls,do_statement_packing(iter));
}

/* a packer takes a list of statement list and generate a list of statement*/
typedef list (*packer)(list);

static list packer_keep_ordering(list statements)
{
    list out = NIL;
    FOREACH(LIST,l,statements)
    {
        gen_sort_list(l,compare_statements);
        out=gen_nconc(out,l);
    }
    return out;
}

static packer select_statement()
{
    /* will use properties later */
    return packer_keep_ordering;
}

static void pack_sequence(sequence seq,graph dependence_graph)
{
    /* we will use successors */
    init_statement_successors_map(sequence_statements(seq),dependence_graph);

    /* do the packing */
    list packs = statement_packing(sequence_statements(seq));
    list pack = select_statement()(packs);
    sequence_statements(seq)=pack;
   
    /* and the cleaning */
    free_statement_successors_map();
}

bool pack_statements(const char * module_name)
{
    /* get the resources */
    set_current_module_statement( (statement)db_get_memory_resource(DBR_CODE, module_name,true)); 
    set_current_module_entity(module_name_to_entity(module_name));
	set_ordering_to_statement(get_current_module_statement());
    graph dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    /* Now do the job */

    gen_context_recurse(get_current_module_statement(), dependence_graph,sequence_domain,
            gen_true, pack_sequence);

    pips_assert("Statement is consistent after packing", 
            statement_consistent_p(get_current_module_statement()));

    /* Reorder the module, because statements have been moved */
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    /* update/release resources */
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    return true;
}

#endif
bool pack_statements(const char * module_name)  { return true ; }
