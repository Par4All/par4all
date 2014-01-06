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

/* Name     : single_assign.c
 * Package  : reindexing
 * Author   : Alexis Platonoff
 * Date     : febuary 1994
 * Historic :
 *   - 14 nov 94 : move this file into reindexing package. Del single_assign
 *     package.
 *
 * Documents: 
 * Comments : This file contains the functions for the transformation of a
 * program to a single assignment form.
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrice.h"
#include "matrix.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pip.h"
#include "array_dfg.h"
#include "reindexing.h"

/* Macro functions  	*/

#define FIRST 1
#define SECOND 2
#define THIRD 3

/* Internal variables 	*/

/* Global variables */
static int tc;

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

/* ========================================================================= */
/* * void rhs_subs_in_ins(instruction ins, reference r1, reference r2)
 *
 * Changes, in the right hand side of "ins", the occurrences of "r1" by
 * "r2".
 * 
 */
void rhs_subs_in_ins(ins, r1, r2)
instruction ins;
reference r1, r2;
{
    switch(instruction_tag(ins)) {
      case is_instruction_call : {
	  call c = instruction_call(ins);
	  if(ENTITY_ASSIGN_P(call_function(c))) {
	    expression rhs_exp = EXPRESSION(CAR(CDR(call_arguments(c))));

	    ref_subs_in_exp(rhs_exp, r1, r2);

	  }
	  else pips_internal_error("Instruction is not an assign call");
	  
	  break;
    }
    case is_instruction_block :
    case is_instruction_test :
    case is_instruction_loop :
    case is_instruction_goto :
    case is_instruction_unstructured :
    default : pips_internal_error("Instruction is not an assign call");
  }
}

/*==================================================================*/
/* vertex  my_dfg_in_vertex_list( (list) l, (vertex) v )
 * Input  : A list l of vertices.
 *          A vertex v of a dataflow graph.
 * Returns vertex_undefined if v is not in list l.
 * Returns v' that has the same statement_ordering than v.
 *
 * AC 93/10/19
 */

vertex my_dfg_in_vertex_list( l, v )

 list       l;
 vertex     v;
{
 vertex     ver;
 int        in;

 in = dfg_vertex_label_statement(vertex_vertex_label(v));
 for (;!ENDP(l); POP(l))
    {
     int prov_i;

     ver = VERTEX(CAR( l ));
     prov_i = dfg_vertex_label_statement(vertex_vertex_label(ver));
     if ( prov_i == in ) return( ver );
    }
 return (vertex_undefined);
}


/*=======================================================================*/
/* graph my_dfg_reverse_graph( (graph) g )
 * This function is used to reverse Pips's graph in order to have
 * all possible sources directly (Feautrier's dependance graph).
 *
 * AC 93/10/19
 */


graph my_dfg_reverse_graph( g )

 graph       g;
{
 graph       rev_graph = graph_undefined;
 list        verlist = NIL;
 successor   succ;

 MAPL(ver_ptr,{
      vertex          ver;
      vertex          ver2;
      vertex          ver5;

      ver  = VERTEX(CAR( ver_ptr ));
      ver5 = my_dfg_in_vertex_list( verlist, ver );
      if ( ver5 == vertex_undefined )
         {
/*
          ver2 = make_vertex(copy_dfg_vertex_label((dfg_vertex_label)\
                             vertex_vertex_label( ver)),(list) NIL );
*/
          ver2 = make_vertex(vertex_vertex_label(ver), NIL);
          ADD_ELEMENT_TO_LIST( verlist, VERTEX,ver2);
         }
      else ver2 = ver5;

      MAPL(succ_ptr, {
           list            li = NIL;
           successor       succ2;
           vertex          ver3;
           vertex          ver4;

           succ  = SUCCESSOR(CAR( succ_ptr ));
           ver3  = successor_vertex( succ );
           ver5  = my_dfg_in_vertex_list( verlist, ver3);
/*
           succ2 = make_successor(copy_dfg_arc_label((dfg_arc_label)\
                                     successor_arc_label(succ)),ver2 );
*/
           succ2 = make_successor(successor_arc_label(succ), ver2);
           if ( ver5 == vertex_undefined )
              {
               ADD_ELEMENT_TO_LIST( li, SUCCESSOR,succ2);
/*
               ver4 = make_vertex(copy_dfg_vertex_label((dfg_vertex_label)\
                                  vertex_vertex_label(ver3)),(list) li );
*/
               ver4 = make_vertex(vertex_vertex_label(ver3), li);
               ADD_ELEMENT_TO_LIST( verlist,VERTEX,ver4);
              }
           else
             ADD_ELEMENT_TO_LIST(vertex_successors(ver5), SUCCESSOR, succ2);
          }, vertex_successors( ver ) );


     }, graph_vertices( g ) );

 rev_graph = make_graph( verlist );
 return( rev_graph );
}

/* ========================================================================= */
/*
 * entity create_entity(string name, variable v)
 * 
 * Creates an entity with the name "name" and of variable "v" (i.e. it is
 * its type).
 */
entity create_entity(name, v)
string name;
variable v;
{
  return(make_entity(name,
		     make_type(is_type_variable, v),
		     make_storage(is_storage_rom, UU),
		     make_value(is_value_unknown, UU)));
}


/* ========================================================================= */
/*
 * list dims_of_nest(int n)
 *
 * returns a list containing the dimensions (NewGen dimension) of the
 * englobing loops of statement number "n".
 */
list dims_of_nest(n)
int n;
{
  static_control stco = get_stco_from_current_map(adg_number_to_statement(n));
  list dims = NIL, l_loops;

  l_loops = static_control_loops(stco);
  for(; !ENDP(l_loops); POP(l_loops)) {
    loop lo = LOOP(CAR(l_loops));
    range ra = loop_range(lo);
    dims = gen_nconc(dims, CONS(DIMENSION,
				make_dimension(range_lower(ra),
					       range_upper(ra)), NIL));
  }
  return(dims);
}


/* ========================================================================= */
/*
 * reference build_new_ref(int kind, int n, list subscripts, reference old_r)
 * 
 * builds a new array reference. Its entity name depends on "kind":
 *	kind == IS_TEMP => name is : SATn
 *	kind == IS_NEW_ARRAY => name is : SAIn
 * We first test if this entity does not exist yet. If not, we have create it
 * with a type_variable with the same basic as the one of the entity of
 * "old_ref" and a dimension depending again on kind:
 *	kind == IS_TEMP => dimension is: empty
 *	kind == IS_NEW_ARRAY => dimension is: dimension of the loop nest of
 *					      statement number n
 *
 * Its indices of the new reference are "subscripts".
 *
 * "subscripts" is a list of affine expressions.
 * If "subscripts" is empty, then this is a scalar.
 */
reference build_new_ref(kind, n, subscripts, old_r)
int kind;
int n;
list subscripts;
reference old_r;
{
  list sl;
  entity ent;
  string num, name = (string) NULL;

  /* we duplicate this list */
  sl = subscripts;

  num = (string) malloc(32);
  (void) sprintf(num, "%d", n);
  if(kind == IS_TEMP)
    name = (string) strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
				       SAT, num, (string) NULL));
  else if(kind == IS_NEW_ARRAY)
    name = (string) strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
				       SAI, num, (string) NULL));
  else
    pips_internal_error("Bad value for kind");

  ent = gen_find_tabulated(name, entity_domain);
  if(ent == entity_undefined) {
    list dims = NIL;

    if(kind == IS_NEW_ARRAY)
      dims = dims_of_nest(n);
    else
      pips_internal_error("Bad value for kind");

    ent = create_entity(name, make_variable(basic_of_reference(old_r), dims));
  }

if(get_debug_level() > 6) {
fprintf(stdout, "\t\t\t\t\t\t[build_new_ref] Nouvelle ref %s[", entity_local_name(ent));
fprint_list_of_exp(stdout, sl);
fprintf(stdout, "]\n");
}

  return(make_reference(ent, sl));
}

/* ========================================================================= */
/*
 * void lhs_subs_in_ins(instruction ins, string SA, int n, list subscripts)
 * 
 * Substitutes to the lhs (left Hand Side) reference of "ins" the array
 * reference SAn[subscripts], cf. build_new_ref().
 *
 * "subscripts" is a list of entity, so we have transform it into a list of
 * expression.
 *
 * Note: "ins" must be an assign call
 */
void lhs_subs_in_ins(ins, SA, n, subscripts)
instruction ins;
string SA;
int n;
list subscripts;
{
  switch(instruction_tag(ins)) {
    case is_instruction_call : {
      call c = instruction_call(ins);
      if(ENTITY_ASSIGN_P(call_function(c))) {
	expression lhs_exp = EXPRESSION(CAR(call_arguments(c)));
	syntax sy = expression_syntax(lhs_exp);
	if(syntax_reference_p(sy)) {
	  reference lhs = syntax_reference(sy);
	  list exp_subs = entities_to_expressions(subscripts);
	  syntax_reference(sy) = build_new_ref(IS_NEW_ARRAY, n, exp_subs, lhs);

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\t[lhs_subs_in_ins] New ref %s instead of %s\n",
	words_to_string(words_reference(syntax_reference(sy))),
	words_to_string(words_reference(lhs)));
}

	}
	else pips_internal_error("Lhs is not a reference");
      }
      else pips_internal_error("Instruction is not an assign call");
      break;
    }
    case is_instruction_block :
    case is_instruction_test :
    case is_instruction_loop :
    case is_instruction_goto :
    case is_instruction_unstructured :
    default : pips_internal_error("Instruction is not an assign call");
  }
}

/* ========================================================================= */
/*
 * list references_of_expression(expression exp)
 * 
 * Conversion of an expression into a list of references.
 * Only the array and scalar references used in the operations are added,
 * not the references in index expressions.
 */
list references_of_expression(exp)
expression exp;
{
  list refl = NIL;
  syntax sy = expression_syntax(exp);
  switch(syntax_tag(sy)) {
    case is_syntax_reference: {
      refl = gen_nconc(refl, CONS(REFERENCE, syntax_reference(sy), NIL));
      break;
    }
    case is_syntax_call: {
      list ael = call_arguments(syntax_call(sy));
      for(; !ENDP(ael); POP(ael)) {
	expression ae = EXPRESSION(CAR(ael));
        refl = gen_nconc(refl, references_of_expression(ae));
      }
      break;
    }
    case is_syntax_range: pips_internal_error("Syntax Range");
    default : pips_internal_error("Bad syntax tag");
  }
  return(refl);
}


/* ========================================================================= */
/*
 * list get_rhs_of_instruction(instruction ins)
 * 
 * Constructs a list of references that are all the rhs (Right Hand Side) of
 * instruction "ins".
 */
list get_rhs_of_instruction(ins)
instruction ins;
{
  list rhsl = NIL;
  switch(instruction_tag(ins)) {
    case is_instruction_call : {
      call c = instruction_call(ins);
      if(ENTITY_ASSIGN_P(call_function(c))) {
        expression rhs_exp;
	list args = call_arguments(c);

	if(gen_length(args) != 2)
	  pips_internal_error("Assign call without 2 args");

        /* There are two args: lhs = rhs, we want the references of the rhs */
	rhs_exp = EXPRESSION(CAR(CDR(args)));
	rhsl = gen_nconc(rhsl, references_of_expression(rhs_exp));
      }
      break;
    }
    case is_instruction_block :
    case is_instruction_test :
    case is_instruction_loop :
    case is_instruction_goto :
    case is_instruction_unstructured :
    default : pips_internal_error("Instruction is not an assign call");
  }
  return(rhsl);
}


/* ========================================================================= */
/*
 * list build_associate_temp(list lr)
 *
 * builds a list of reference to temporary variables. These temporaries are
 * associated to each reference of "lr" (in the same order). These temporaries
 * are scalars.
 * They are numbered with a global counter "tc".
 * 
 */
list build_associate_temp(lr)
list lr;
{
  extern int tc;

  list l, atl = NIL;

  for(l = lr; !ENDP(l); POP(l)) {
    reference r = REFERENCE(CAR(l));
    atl = gen_nconc(atl, CONS(REFERENCE,
			      build_new_ref(IS_TEMP, tc++, NIL, r),
			      NIL));
  }
  return(atl);
}


/* ========================================================================= */
/*
 * list build_successors_with_rhs(list ls, reference r)
 *
 * builds a sublist of list "ls". "ls" is a list of successors, each successor
 * containing a list of dataflows. Each datalow represents the dependence
 * upon a reference. So a successor of "ls" is put in our sublist if and only
 * if it contains a dataflow that depends on the reference "r".
 * 
 */
list build_successors_with_rhs(ls, r)
list ls;
reference r;
{
  list sls = NIL, l, ll;

  for(l = ls; !ENDP(l); POP(l)) {
    successor suc = SUCCESSOR(CAR(l));
    list dfl = dfg_arc_label_dataflows(successor_arc_label(suc));
    for(ll = dfl; !ENDP(ll); POP(ll)) {
      dataflow df = DATAFLOW(CAR(ll));
      reference dfr = dataflow_reference(df);

      /* true equality, not on pointers */
      if(reference_equal_p(dfr, r))
	sls = gen_nconc(sls, CONS(SUCCESSOR, suc, NIL));
    }
  }
  return(sls);
}


/* ========================================================================= */
/*
 * int count_dataflows_on_ref(list ls, reference r, dataflow *df, int *m)
 *
 * "ls" is a list of successors. This function scans this list and looks for
 * dataflows on reference "r". If it does not find any, it returns 0, else
 * if there is one such dataflow, then it returns 1 (and also, save this
 * dataflow and the statement number of the pointed vertex in, respectively,
 * df and m). Else, it returns 2.
 * 
 */
int count_dataflows_on_ref(ls, r, df, m)
list ls;
reference r;
dataflow *df;
int *m;
{
  list l;
  int count = 0;

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\t[count_dataflows_on_ref] %s\n",
	words_to_string(words_reference(r)));
}

  *df = dataflow_undefined;
  *m = -1;

  for(l = ls; !ENDP(l) && (count < 2); POP(l)) {
    successor suc = SUCCESSOR(CAR(l));
    list dfl = dfg_arc_label_dataflows(successor_arc_label(suc));
    for(; !ENDP(dfl) && (count < 2); POP(dfl)) {
      dataflow d = DATAFLOW(CAR(dfl));

      if(reference_equal_p(r, dataflow_reference(d))) {
	count++;
        if(count == 2)
	  *df = dataflow_undefined;
	else {
	  *df = d;
	  *m = dfg_vertex_label_statement(vertex_vertex_label(successor_vertex(suc)));

	}
	if(get_debug_level() > 3) {
	  fprintf(stdout, "\t\t\t[count_dataflows_on_ref] One more\n");
	  fprint_dataflow(stdout,
			  dfg_vertex_label_statement(vertex_vertex_label(successor_vertex(suc))),
			  d);
}
      }
    }
  }
  return(count);
}


/* ========================================================================= */
/*
 * list dataflows_on_ref(successor suc, reference r)
 *
 * This function scans the list of dataflows of "suc" and looks for
 * dataflows on reference "r". It returns the list these dataflows.
 */
list dataflows_on_ref(suc, r)
successor suc;
reference r;
{
  list dfl, rdfl = NIL;

  dfl = dfg_arc_label_dataflows(successor_arc_label(suc));
  for(; !ENDP(dfl); POP(dfl)) {
    dataflow d = DATAFLOW(CAR(dfl));
    if(reference_equal_p(r, dataflow_reference(d)))
      rdfl = gen_nconc(rdfl, CONS(DATAFLOW, d, NIL));
  }
  return(rdfl);
}


/* ========================================================================= */
/*
 * void ref_subs_in_exp(expression exp, reference r1, reference r2)
 *
 * changes, in expression "exp", the occurrences of "r1" by "r2".
 * 
 */
void ref_subs_in_exp(exp, r1, r2)
expression exp;
reference r1, r2;
{
  syntax sy = expression_syntax(exp);
  switch(syntax_tag(sy)) {
    case is_syntax_reference: {
      if(reference_equal_p(r1, syntax_reference(sy)))
	syntax_reference(sy) = r2;
      break;
    }
    case is_syntax_call: {
      list ael = call_arguments(syntax_call(sy));
      for(; !ENDP(ael); POP(ael)) {
	expression ae = EXPRESSION(CAR(ael));
        ref_subs_in_exp(ae, r1, r2);
      }
      break;
    }
    case is_syntax_range: pips_internal_error("Syntax Range");
    default : pips_internal_error("Bad syntax tag");
  }
}

/* ========================================================================= */
/*
 * expression predicate_to_expression(predicate pred)
 */
expression predicate_to_expression(pred)
predicate pred;
{
  entity and_ent, leq_ent, equ_ent;
  expression exp1 = expression_undefined, exp2;
  Psysteme ps = (Psysteme) predicate_system(pred);
  Pcontrainte pc;

if(get_debug_level() > 5) {
fprintf(stdout, "\t\t\t\t\t[predicate_to_expression] Init\n");
fprint_pred(stdout, pred);
}

  and_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						    AND_OPERATOR_NAME),
			       entity_domain);
  leq_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						    LESS_OR_EQUAL_OPERATOR_NAME),
			       entity_domain);
  equ_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						    EQUAL_OPERATOR_NAME),
			       entity_domain);

  if( (and_ent == entity_undefined) || (leq_ent == entity_undefined) ||
      (equ_ent == entity_undefined) ) {
    pips_internal_error("There is no entity for operators");
  }

  for(pc = ps->inegalites; pc!=NULL; pc = pc->succ) {
    Pvecteur pv = pc->vecteur;
    expression exp = make_vecteur_expression(pv);
    exp2 = MakeBinaryCall(leq_ent, exp, int_to_expression(0));

if(get_debug_level() > 6) {
pu_vect_fprint(stdout, pv);
fprintf(stdout, "\t\t\t\t\t\tvec_to_exp : %s\n",
	words_to_string(words_expression(exp)));

fprintf(stdout, "\t\t\t\t\t\tINEG: exp2 = %s\n",
	words_to_string(words_expression(exp2)));
}

    if(exp1 == expression_undefined)
      exp1 = exp2;
    else
      exp1 = MakeBinaryCall(and_ent, exp1, exp2);

if(get_debug_level() > 6) {
fprintf(stdout, "\t\t\t\t\t\tINEG: exp1 = %s\n",
	words_to_string(words_expression(exp1)));
}

  }
  for(pc = ps->egalites; pc!=NULL; pc = pc->succ) {
    Pvecteur pv = pc->vecteur;
    exp2 = MakeBinaryCall(equ_ent, make_vecteur_expression(pv),
    			  int_to_expression(0));
    if(exp1 == expression_undefined)
      exp1 = exp2;
    else
      exp1 = MakeBinaryCall(and_ent, exp1, exp2);

if(get_debug_level() > 6) {
fprintf(stdout, "\t\t\t\t\t\tEG: exp1 = %s, exp2 = %s\n",
	words_to_string(words_expression(exp1)),
	words_to_string(words_expression(exp2)));
}
  }

if(get_debug_level() > 5) {
fprintf(stdout, "\t\t\t\t\t[predicate_to_expression] Result: %s\n",
	words_to_string(words_expression(exp1)));
}

  return(exp1);
}


/* ========================================================================= */
/*
 * void add_test(statement s, predicate cond, reference lhs rhs, int how)
 * 
 * "s" is a statement that contains a block instruction, i.e. a list of
 * instruction. This function adds a new assignment statement to this list
 * conditionned by "cond". This assignment is "lhs = rhs". The kind of test
 * instruction is determined by "how".
 *
 * "how" can have three different values:
 *	_ FIRST: adds a test instruction in the before last position of this
 * 	list
 *	_ SECOND: this case implies that the before last instruction of this
 *	list is a test, it adds a "else if" instruction
 *	_THIRD: this case implies that the before last instruction of this
 *      list is a test, it adds a "else" instruction
 *
 */
void add_test(s, cond, lhs, rhs, how)
statement s;
predicate cond;
reference lhs, rhs;
int how;
{
  list bl, l, ll = NIL, lll = NIL;
  entity assign_ent;
  expression lhs_exp, rhs_exp, pred_exp;
  statement assign_s, new_s;
  instruction ins = statement_instruction(s);

  if(instruction_tag(ins) != is_instruction_block)
    pips_internal_error("Instruction MUST be a block");

  /* We construct the statement "lhs = rhs" */
  assign_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
  						       ASSIGN_OPERATOR_NAME),
				  entity_domain);
  lhs_exp = make_expression(make_syntax(is_syntax_reference, lhs), normalized_undefined);
  rhs_exp = make_expression(make_syntax(is_syntax_reference, rhs), normalized_undefined);
  assign_s = make_statement(entity_empty_label(), STATEMENT_NUMBER_UNDEFINED,
  			    STATEMENT_ORDERING_UNDEFINED, string_undefined,
			    make_instruction(is_instruction_call,
					     make_call(assign_ent,
					               CONS(EXPRESSION, lhs_exp,
							    CONS(EXPRESSION,
								 rhs_exp, NIL)))));

  /* Then, we find where to put it... */
  bl = instruction_block(ins);
  for(l = bl; !ENDP(l); POP(l)) {lll = ll; ll = l;}
  if(ll == NIL) /* This means that "bl" is empty */
    pips_internal_error("Block is empty");

  /* ... and how. */
  if(how == FIRST) {
    pred_exp = predicate_to_expression(cond);
    new_s = make_statement(entity_empty_label(), 
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED, string_undefined,
		           make_instruction(is_instruction_test,
					    make_test(pred_exp,
						      assign_s,
						      make_empty_statement())));

    if(lll == NIL) /* This means that "bl" has only one instruction */
      instruction_block(ins) = CONS(STATEMENT, new_s, bl);
    else
      CDR(lll) = CONS(STATEMENT, new_s, ll);
  }
  else if(how == SECOND) {
    pred_exp = predicate_to_expression(cond);
    if(lll == NIL)
      pips_internal_error("Block has only one instruction");
    else {
      instruction ai;
      test te;

      new_s = STATEMENT(CAR(lll));
      ai = statement_instruction(new_s);
      if(instruction_tag(ai) != is_instruction_test)
	pips_internal_error("Instruction must be a test");

      te = instruction_test(ai);
      while(test_false(te) != statement_undefined) {
	instruction aai = statement_instruction(test_false(te));
	if(instruction_tag(aai) != is_instruction_test)
	  pips_internal_error("Instruction must be a test");
	te = instruction_test(aai);
      }
      test_false(te) = make_statement(entity_empty_label(),
				      STATEMENT_NUMBER_UNDEFINED,
				      STATEMENT_ORDERING_UNDEFINED, string_undefined,
				      make_instruction(is_instruction_test,
						       make_test(pred_exp,
								 assign_s,
								 make_empty_statement())));
    }
  }
  else if(how == THIRD) {
    if(lll == NIL)
      pips_internal_error("Block has only one instruction");
    else {
      instruction ai;
      test te;

      new_s = STATEMENT(CAR(lll));
      ai = statement_instruction(new_s);
      if(instruction_tag(ai) != is_instruction_test)
	pips_internal_error("Instruction must be a test");

      te = instruction_test(ai);
      while(test_false(te) != statement_undefined) {
	instruction aai = statement_instruction(test_false(te));
	if(instruction_tag(aai) != is_instruction_test)
	  pips_internal_error("Instruction must be a test");
	te = instruction_test(aai);
      }
      test_false(te) = assign_s;
    }
  }
  else /* bad value */
    pips_internal_error("Bad value in how");
}


/* ========================================================================= */
/*
 * void sa_print_ins(FILE *fp, instruction i)
 * 
 */
void sa_print_ins(fp, i)
FILE *fp;
instruction i;
{
  switch(instruction_tag(i)) {
    case is_instruction_block: {
      list l = instruction_block(i);
      fprintf(fp, "Block {\n");
      for(; !ENDP(l); POP(l)) {
	sa_print_ins(fp, statement_instruction(STATEMENT(CAR(l))));
      }
      fprintf(fp, "} End Block\n");
      break;
    }
    case is_instruction_test: {
      test te = instruction_test(i);
      fprintf(fp, "If (%s) {\n",
	      words_to_string(words_expression(test_condition(te))));
      sa_print_ins(fp, statement_instruction(test_true(te)));
      fprintf(fp, "}\n");
      if(test_false(te) != statement_undefined) {
        fprintf(fp, "Else {\n");
        sa_print_ins(fp, statement_instruction(test_false(te)));
        fprintf(fp, "}\n");
      }
      break;
    }
    case is_instruction_loop: {
      loop lo = instruction_loop(i);
      fprintf(fp, "For %s = %s, %s, %s {\n",
      	      entity_local_name(loop_index(lo)),
      	      words_to_string(words_expression(range_lower(loop_range(lo)))),
	      words_to_string(words_expression(range_upper(loop_range(lo)))),
	      words_to_string(words_expression(range_increment(loop_range(lo)))));
      sa_print_ins(fp, statement_instruction(loop_body(lo)));
      fprintf(fp, "}\n");
      break;
    }
    case is_instruction_goto: {
      fprintf(fp, "GOTO %d\n", statement_ordering(instruction_goto(i)));
      break;
    }
    case is_instruction_call: {
      call ca = instruction_call(i);
      list l = call_arguments(ca);;
      fprintf(fp, "Call %s with:", entity_local_name(call_function(ca)));
      for(; !ENDP(l); POP(l)) {
	fprintf(fp, "%s, ",
		words_to_string(words_expression(EXPRESSION(CAR(l)))));
      }
      fprintf(fp, "\n");
      break;
    }
    case is_instruction_unstructured:
    default: pips_internal_error("Bad instruction tag");
  }
}


/* ========================================================================= */
/*
 * bool full_predicate_p(predicate p)
 */
bool full_predicate_p(p)
predicate p;
{
  Psysteme ps;

  if(p == predicate_undefined) return(true);

  ps = (Psysteme) predicate_system(p);

  if(ps == NULL) return(true);

  if((ps->egalites == NULL) && (ps->inegalites == NULL)) return(true);

  return(false);
}


/* ========================================================================= */
/*
 * void sa_do_it(graph the_dfg)
 * 
 */
void sa_do_it(the_dfg)
graph the_dfg;
{
  list l;

  /* let's have a reverse DFG, i.e. the "successors" of an instruction i are
   * the instructions that may write the values used by i;
   */
  the_dfg = my_dfg_reverse_graph(the_dfg);

if(get_debug_level() > 0) {
fprint_dfg(stdout, the_dfg);
}

  /* loop over the vertices of the DFG */
  for(l = graph_vertices(the_dfg); !ENDP(l); POP(l)) {

    /* cv is the current vertex, cn is its number, cs is the associate
     * statement
     */
    vertex cv = VERTEX(CAR(l));
    int cn = dfg_vertex_label_statement(vertex_vertex_label(cv));
    statement cs = adg_number_to_statement(cn);
    instruction ci = statement_instruction(cs);
    static_control stco = get_stco_from_current_map(adg_number_to_statement(cn));
    list ciel = static_control_to_indices(stco);

    reference new_ref;
    list clrhs, clt;

    /* cls is the list of successors of cv */
    list cls = vertex_successors(cv);

if(get_debug_level() > 0) {
fprintf(stdout, "Noeuds %d :\n", cn);
sa_print_ins(stdout, ci);
fprintf(stdout, "\n");
}

    if(! assignment_statement_p(cs))
      continue;

    /* change the lhs of cs to a new array indexed by the englobing loop
     * indices: SAIn(i1,...ip)
     * (i1,...,ip) are the indices of the englobing loops of cs
     */
    lhs_subs_in_ins(ci, SAI, cn, ciel);

if(get_debug_level() > 1) {
fprintf(stdout, "\tApres Subs LHS : \n");
sa_print_ins(stdout, statement_instruction(cs));
}

    if(cls == NIL)
      continue;

    /* construct the list of rhs */
    clrhs = get_rhs_of_instruction(ci);

    /* construct the associated list of temporaries */
    clt = build_associate_temp(clrhs);

    /* in the code, replace ci by a block new_i containing ci */
    if (instruction_tag(ci) != is_instruction_block) {
      /* We create a new instruction with an empty block of instructions */
      instruction new_i = make_instruction_block(NIL);
      /* Then, we put the instruction "ci" in the block. */
      instruction_block(new_i) = CONS(STATEMENT,
				      make_statement(statement_label(cs),
						     statement_number(cs),
						     statement_ordering(cs),
						     statement_comments(cs),
						     ci),
				      NIL);
      statement_label(cs) = entity_empty_label();
      statement_number(cs) = STATEMENT_NUMBER_UNDEFINED;
      statement_ordering(cs) = STATEMENT_ORDERING_UNDEFINED;
      statement_comments(cs) = string_undefined;
      statement_instruction(cs) = new_i;

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tDevient BLOCK : \n");
sa_print_ins(stdout, statement_instruction(cs));
}
    }

    /* loop over the list of rhs */
    for(; !ENDP(clrhs); POP(clrhs), POP(clt)) {

      /* crhs is the current ref and ct is the associate temp */
      reference crhs = REFERENCE(CAR(clrhs));
      reference ct = REFERENCE(CAR(clt));
      dataflow df;
      int m, nb_df;

if(get_debug_level() > 2) {
fprintf(stdout, "\t\tReference %s :\n",
	words_to_string(words_reference(crhs)));
}

      /* We count the number of dataflows in "cls" that contains "crhs", and
       * we then consider three cases: 0 dataflow, 1 dataflow, 2 or more
       * dataflows.
       */
      nb_df = count_dataflows_on_ref(cls, crhs, &df, &m);

      if(nb_df == 0) {
	/* Nothing is changed concerning this rhs */

if(get_debug_level() > 2) {
fprintf(stdout, "\t\tZero dataflow :\n");
sa_print_ins(stdout, statement_instruction(cs));
}

      }
      else if(nb_df == 1) {
	/* pred is the cond of df, L is the transformation of df */
	predicate pred = dataflow_governing_pred(df);
	list L = dataflow_transformation(df);

if(get_debug_level() > 2) {
fprintf(stdout, "\t\tUn seul dataflow :\n");
sa_print_ins(stdout, statement_instruction(cs));
}

	new_ref = build_new_ref(IS_NEW_ARRAY, m, L, crhs);

	if(full_predicate_p(pred)) {
	  /* change in ci the occurrences of crhs by SAIm[L] */
	  rhs_subs_in_ins(ci, crhs, new_ref);

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tFull predicate :\n");
sa_print_ins(stdout, statement_instruction(cs));
}
	}
	else {
	  rhs_subs_in_ins(ci, crhs, ct);
	  add_test(cs, pred, ct, new_ref, FIRST);
          add_test(cs, predicate_undefined, ct, crhs, THIRD);

if(get_debug_level() > 2) {
fprintf(stdout, "\t\t\tNot full predicate :\n");
sa_print_ins(stdout, statement_instruction(cs));
}
        }
      }
      else {
	int nt = 0; /* number of tests currently added */
	list lls = cls;

if(get_debug_level() > 2) {
fprintf(stdout, "\t\tAu moins deux dataflows :\n");
}
	for(; !ENDP(lls); POP(lls)) {
	  successor suc = SUCCESSOR(CAR(lls));

	  list ldf = dataflows_on_ref(suc, crhs);
	  m = dfg_vertex_label_statement(vertex_vertex_label(successor_vertex(suc)));

if(get_debug_level() > 2) {
fprintf(stdout, "\t\tArc pointe' vers %d :\n", m);
}

	  for(; !ENDP(ldf); POP(ldf)) {
	    dataflow df = DATAFLOW(CAR(ldf));

	    /* pred is the cond of df, L is the transformation of df */
            predicate pred = dataflow_governing_pred(df);
	    list L = dataflow_transformation(df);

            /* change in ci the occurrences of crhs by ct; */
            rhs_subs_in_ins(ci, crhs, ct);

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tApres substitution dans RHS :\n");
sa_print_ins(stdout, statement_instruction(cs));
}

	    new_ref = build_new_ref(IS_NEW_ARRAY, m, L, crhs);

	    /* First test */
	    if(nt == 0) {
	      /* "if(pred) ct = SAIm[L(i1,...,ip)]", just before ci in cs */
	      add_test(cs, pred, ct, new_ref, FIRST);
	      nt++;

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tPremier test :\n");
sa_print_ins(stdout, statement_instruction(cs));
}
	    }
	    else {
	      /* "else if(pred) ct = SAIm[L(i1,...,ip)]", just before ci in cs */
	      add_test(cs, pred, ct, new_ref, SECOND);

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tDeuxieme test :\n");
sa_print_ins(stdout, statement_instruction(cs));
}
	    }
	  }
	}
	/* "else ct = crhs", just before ci in cs */
        add_test(cs, predicate_undefined, ct, crhs, THIRD);

if(get_debug_level() > 3) {
fprintf(stdout, "\t\t\tDernier test :\n");
sa_print_ins(stdout, statement_instruction(cs));
}
      }
    }
  }
}


/* ========================================================================= */
/*
 * void single_assign((char*) mod_name):
 * 
 */
void single_assign(mod_name)
char*   mod_name;
{
  extern int tc;

  graph the_dfg;
  entity ent;
  static_control          stco;
  statement               mod_stat;
  statement_mapping STS;

  /* Initialize debugging functions */
  debug_on("SINGLE_ASSIGN_DEBUG_LEVEL");
  if(get_debug_level() > 0)
    user_log("\n\n *** COMPUTE SINGLE_ASSIGN for %s\n", mod_name);

  /* We get the required data: module entity, code, static_control, dataflow
   * graph.
   */
  ent = local_name_to_top_level_entity( mod_name );

  if(ent != get_current_module_entity()) {
    reset_current_module_entity();
    set_current_module_entity(ent);
  }

  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
  STS = (statement_mapping) db_get_memory_resource(DBR_STATIC_CONTROL,
					    mod_name, true);
  set_current_stco_map(STS);
  stco = get_stco_from_current_map(mod_stat);

  if ( stco == static_control_undefined) {
    pips_internal_error("This is an undefined static control !");
  }
  if ( !static_control_yes( stco )) {
    pips_internal_error("This is not a static control program !");
  }

  /* The DFG */
  the_dfg = adg_pure_dfg((graph) db_get_memory_resource(DBR_ADFG, mod_name, true));

  if(get_debug_level() > 0) {
    fprint_dfg(stdout, the_dfg);
  }

  /* The temporaries counter */
  tc = 0;

  sa_do_it(the_dfg);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), (char*) mod_stat);

  if(get_debug_level() > 0) {
    user_log("\n\n *** SINGLE_ASSIGN done\n");
  }

  reset_current_stco_map();

  debug_off();
}

