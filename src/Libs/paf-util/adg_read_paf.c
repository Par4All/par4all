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

/* Name     : adg_read_paf.c
 * Package  : paf-util
 * Author   : Alexis Platonoff
 * Date     : april 1993
 * Historic : 16 july 93, changes in paf_ri, AP
 *            2 august 93, moved from (package) array_dfg to paf-util, AP
 * Documents:
 *
 * Comments :
 * This functions are used to store a data flow graph in a NEWGEN structure
 * from the reading of three files generate by the PAF parallelizer:
 *
 *              _ the ".src" file, which contains the DFG nodes
 *              _ the ".do" file, which contains the LOOP list
 *              _ the ".gd" file, which contains the DG nodes
 *
 * The DFG nodes include the source and sink instructions, the reference, the
 * transformation and the governing predicate. The LOOP list gives the
 * description of each loop of the program, i.e. the index parameter, the lower
 * and upper bounds expressions and the step expression. The DG nodes give for
 * each instruction, the list of the englobing loops. From the loop description
 * and the list of the englobing loops, we can compute for each instruction the
 * execution domain.
 *
 * This program uses two new NEWGEN data structures: "dfg" (which uses the
 * generic graph structure) and "lisp_expression". They are defined in the
 * file paf_ri.f.tex (see comments on them in this file). It also uses the
 * library of PIPS and its RI data structure.
 *
 * The PAF files are read with YACC and LEX programs. We made one grammar for
 * the three types of files (parse.y) and one characters analyser (scan.l).
 * The "yy" and "YY" prefixes are modified into "adgyy" and "ADGYY".
 *
 * In order two simplify the parser, this parsing requires the concatenation of
 * the three files in one with the extension ".paf". This concatenation must
 * be like this (see the file parse.y for the grammar):
 *              ( .src ) ( .do ) ( .gd )
 *
 * Thus, the parsing has three steps:
 *
 * First, the ".src" file is parsed. During this reading we collect the
 * informations for the DFG. All these informations are stored in the global
 * variables "dfg".
 *
 * Second, we read the ".do" file. All the loops and their description are
 * stored in the global variables "loop_list". This variable is a list of
 * "loop" (from the NEWGEN type defined in the RI of PIPS).
 *
 * Third, we parse the ".gd" file. In this reading, we only keep the englobing
 * loops of each instruction. The update of the DFG execution domain is done
 * during the parsing (each time all the englobing loops for a given
 * instruction are known).
 */

/* Ansi includes	*/
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

/* Newgen includes	*/
#include "genC.h"

/* C3 includes		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrice.h"
#include "matrix.h"

/* Pips includes	*/
#include "boolean.h"
#include "linear.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
/*
typedef void * arc_label;
typedef void * vertex_label;
*/
/* Local typedef, probably to be found in paf_ri */
#include "paf_ri.h"
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
#include "graph.h"
#include "dg.h"
#include "misc.h"
/*#include "paf_ri.h"*/
#include "paf-util.h"

/* Macro functions	*/
#define DOT "."
#define PAF_STRING "paf"
#define INS_NAME_LENGTH 4
#define STMT_TO_STCT_SIZE 100  /* hash table max size */


/* Global variables	*/
/* The "dfg" global variable is the current DFG being computed. Its type is
 * defined in graph.h and paf_ri.h.
 */
graph dfg;

/* The "stmt_list" global variable is the list the assign statement of the
 * program (with all fields empty but two: ordering (the number of the
 * statement) and comments (the string name of the statement)).
 */
list stmt_list;

/* The "loop_list" global variable is the list the loops of the program
 * (with all their characteristics: index, bounds, step). The loop type is
 * "loop" defined in ri.h.
 */
list loop_list;

/* The "STS" global variable is the hash table that maps the
 * static_control on the statements.
 */
static hash_table STS;


/* Internal variables 	*/
static vertex crt_node;		/* Current source node */
static int sink_stmt,		/* Current sink statement */
	   source_stmt,		/* Current source statement */
	   crt_stmt;		/* Current stmt (an integer) */
static reference ref;		/* Current reference */
static expression crt_exp;	/* Current expression */
static predicate gov_pred,	/* Current governing predicate */
		 exec_dom;	/* Current execution domain */
static list crt_node_l,		/* Current list of nodes */
	    trans_l,		/* Current list of transformations */
	    ref_inds,		/* Current list of reference indices */
	    lin_exp_l,		/* Current list of linear expressions */
	    pred_l,		/* Current list of predicates */
	    crt_el,		/* Current list of englobing loops */
	    param_l;		/* Current list of structure parameters */
static string crt_op_name;	/* Current operator name */
static loop crt_loop;		/* Current loop instruction */

/* Local typedef */
/*
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
*/

/*============================================================================*/
/* void adg_read_paf(char * s) :
 *
 * computes the DFG of the PAF program name given in argument and returns
 * it.
 *
 * The global variables "loop_list" and "stmt_list" are initialized to NIL.
 *
 * The DFG is put in the variable "dfg", which is the value returned.
 *
 * Note : This function creates a statement_mapping (i.e. a hash_table)
 * which is initialized in this function. This is done by
 * set_current_stco_map(), see paf-util/utils.c. To use this hash_table,
 * you do not have to know its name, only call get_current_stco_map()
 * which returns it. */
graph adg_read_paf(s)
char * s;
{
 extern list loop_list, stmt_list;

 FILE *paf_file;
 char *paf_file_name;

 loop_list = NIL;
 stmt_list = NIL;

 STS = hash_table_make(hash_int, STMT_TO_STCT_SIZE);

 paf_file_name = strdup(concatenate(s, DOT, PAF_STRING, (char *) NULL));

 if( (paf_file = fopen(paf_file_name, "r")) == NULL)
   {
     fprintf(stderr, "Cannot open file %s\n", paf_file_name);
     exit(1);
   }

#if defined(HAS_ADGYY)

 adgyyin = paf_file;
 (void) adgyyparse();

#else

 pips_internal_error("not adgyy{in,parse} compiled in (HAS_ADGYY undef)");

#endif

 fclose(paf_file);

 set_current_stco_map(STS);

 return(dfg);
}


#define INIT_STATEMENT_SIZE 20

/*============================================================================*/
/* void init_new_dfg() : initializes the computation of the DFG, i.e. the
 * creation of the DFG in the "dfg" variable and the initialization of the
 * global variables "exec_dom" and "gov_pred".
 */
void init_new_dfg()
{
 extern predicate exec_dom, gov_pred;
 extern list param_l;

 dfg = make_graph(NIL); /* The list of vertices is empty at the beginning */

 param_l = NIL;
 exec_dom = predicate_undefined;
 gov_pred = predicate_undefined;
}


/*============================================================================*/
/* void new_param(s) : adds a new structure parameters to the global list
 * "param_l".
 */
void new_param(s)
string s;
{
 extern list param_l;

 entity new_ent;
 string param_full_name;

 param_full_name = strdup(concatenate(DFG_MODULE_NAME, MODULE_SEP_STRING,
				      s, (char *) NULL));

 new_ent = gen_find_tabulated(param_full_name, entity_domain);

 if(new_ent == entity_undefined)
   /* Fi: UU is not a proper argument for make_basic_int() */
    new_ent = make_entity(param_full_name,
			  make_type_variable(
					     make_variable(make_basic_int(4 /* UU */),
						  NIL, NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value(is_value_unknown, UU));

 param_l = CONS(ENTITY, new_ent, param_l);
}


/*============================================================================*/
/* void init_new_df_sink_ins(): initializes the computation of the sink
 * statement of a datadflow. The structure of the file is such that all
 * dependences with the same sink statement are grouped together. As our graph
 * structure groups the dependences on the same source statement, we have to
 * create a node for each dependence found for this sink statement. All these
 * nodes are kept in "crt_node_l". The sink statement will be known when all
 * these nodes will be computed, that's why they are kept in a special list.
 */
void init_new_df_sink_ins()
{
 crt_node_l = NIL;

 sink_stmt = -1;
 crt_node = vertex_undefined;
 ref = reference_undefined;
}


/*============================================================================*/
/* static statement find_stmt_with_num(int n): returns the statement that has
 * its ordering equal to "n".
 *
 * This computation is done using the global variable "stmt_list" that contains
 * the list of all the statements.
 */
static statement find_stmt_with_num(n)
int n;
{
 extern list stmt_list;

 list aux_l = stmt_list;

 for(; aux_l != NIL; aux_l = CDR(aux_l))
   {
    statement aux_stmt = STATEMENT(CAR(aux_l));
    int stmt_order = statement_ordering(aux_stmt);
    if(stmt_order == n)
       return(aux_stmt);
   }
 return(statement_undefined);
}


/*============================================================================*/
/* void init_new_df_source(char *s_ins): initializes the computation of
 * the source statement of a datadflow. This statement is represented by its
 * ordering contained in its name "s_ins". We initialize the list of
 * transformations that will be associated with source statement ("trans_l").
 * Also, we initialize "lin_exp_l" which is used for the parsing of the lisp
 * expressions.
 *
 * Note: We don't forget to update the list of statements "stmt_list".
 */
void init_new_df_source(s_ins)
char * s_ins;
{
 extern list trans_l, stmt_list;

 trans_l = NIL;

 /* In PAF, an statement name is a string "ins_#", where "#" is the number
  * associated with the statement. We get this number.
  */
 source_stmt = atoi(strdup(s_ins + INS_NAME_LENGTH));

 /* We update the global list of statements */
 if(find_stmt_with_num(source_stmt) == statement_undefined)
    stmt_list = CONS(STATEMENT,
		     make_statement(entity_undefined, 1, source_stmt,
				    strdup(s_ins), instruction_undefined,
				NIL, // No local declarations
				NULL, // null or empty string...
				empty_extensions (), make_synchronization_none()),
		     stmt_list);

/* Initialization of global variables */
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void new_df_trans_exp(): The parser has now completed the reading of
 * one transformation expression. We update "trans_l" and reinitialize
 * "lin_exp_l" for the next expression.
 */
void new_df_trans_exp()
{
 if(crt_exp == expression_undefined)
    pips_internal_error("current expression is undefined");

 trans_l = gen_nconc(trans_l, CONS(EXPRESSION, crt_exp, NIL));
 crt_exp = expression_undefined;

/* Initialization of global variables */
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void finish_new_df_source(): The reading of source of the current
 * dataflow is completed. We create this dataflow, the successor to which it is
 * attached and the node from which the dataflow comes. This node is put in the
 * global list "crt_node_l" that contains all the computed edges that have
 * the same sink statement (which is still not known).
 *
 * At this time of the computation, only the source statement and the list of
 * transformations are known. The governing predicate will be added on all
 * the node of "crt_node_l", as well as the reference and the execution
 * domain will be computed afterwards, when all the graph will be made.
 */
void finish_new_df_source()
{
 extern predicate exec_dom;

 successor source_succ;
 list crt_df;

 crt_df = CONS(DATAFLOW, make_dataflow(ref, trans_l, gov_pred,
				       communication_undefined), NIL);
 source_succ = make_successor(make_dfg_arc_label(crt_df),
			      vertex_undefined);

 crt_node = make_vertex(make_dfg_vertex_label(source_stmt, exec_dom,
					      sccflags_undefined),
                        CONS(SUCCESSOR, source_succ, NIL));

 crt_node_l = gen_nconc(crt_node_l, CONS(VERTEX, crt_node, NIL));
}


/*============================================================================*/
/* void init_new_df_gov_pred(): Initializes the computation of the
 * governing predicate.
 *
 * For this, we initialize "pred_l" (expressions that will be contained in the
 * predicate) and "lin_exp_l" (for the parsing of each expression).
 */
void init_new_df_gov_pred()
{
 gov_pred = predicate_undefined;
 pred_l = NIL;
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void save_pred(int option): computes one expression of the predicate.
 * Each expression is used twice ; indeed, an expression may be greater or
 * equal than zero (>=) and smaller than zero (<). "option" says in which case
 * we are: POSITIVE indicates that the predicate is >=, with NEGATIVE it is <.
 * However, the C3 library always represents its inequalities with <=. So, the
 * inequality "A >= 0" becomes "-A <= 0" and "A < 0" becomes "A + 1 <= 0".
 *
 * This function updates the global list "pred_l" that contains the current
 * list of predicates. When a new predicate expression is parsed, the POSITIVE
 * is always considered first (that is why only in that case we use "crt_exp").
 * When the NEGATICE case is considered, the corresponding expression (used in
 * the POSITIVE case) is the first expression of the list "pred_l". So, we only
 * have to replace this expression by it equivalent for the NEGATIVE case (that
 * is why the expression is multiplied by -1).
 */
void save_pred(option)
int option;
{
 expression aux_pred;

 if(option == POSITIVE)
   {
    if(crt_exp == expression_undefined)
       pips_internal_error("current expression is undefined");

    /* "A >= 0" becomes "-A <= 0"*/
    pred_l = CONS(EXPRESSION, negate_expression(crt_exp), pred_l);
    crt_exp = expression_undefined;
   }
 else
 /* option == NEGATIVE */
   {
    /* "A < 0" becomes "A + 1 <= 0"*/
    aux_pred = make_op_exp(PLUS_OPERATOR_NAME,
			   negate_expression(EXPRESSION(CAR(pred_l))),
			   int_to_expression(1));

    pred_l = CONS(EXPRESSION, aux_pred, CDR(pred_l));
   }

/* Initialization of global variables */
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void elim_last_pred(): When POSITIVE and NEGATIVE cases of one predicate
 * have been completed, we eliminate the corresponding expression which is the
 * first one of the list "pred_l".
 */
void elim_last_pred()
{
 pred_l = CDR(pred_l);
}


/*============================================================================*/
/* void new_df_gov_pred(): the parser has found the all predicate of the
 * dataflow of the current node, we have to compute it. This predicate is
 * formed with the list of expressions of "pred_l". The function
 * expressions_to_predicate() translates a list of expressions into a
 * predicate.
 */
void new_df_gov_pred()
{
 dataflow df;

 gov_pred = expressions_to_predicate(pred_l);

 df = first_df_of_succ(first_succ_of_vertex(crt_node));
 dataflow_governing_pred(df) = gov_pred;
}


/*============================================================================*/
/* void init_new_df_ref(ichar *s_ref): the parser has gotten the name of
 * the reference on which the current dataflow depends. We compute it and
 * update the global variable "ref".
 */
void init_new_df_ref(s_ref)
char * s_ref;
{
 entity ent_ref;
 string ref_full_name;

 ref_inds = NIL;

 ref_full_name = strdup(concatenate(DFG_MODULE_NAME, MODULE_SEP_STRING,
				    s_ref, (char *) NULL));

 ent_ref = gen_find_tabulated(ref_full_name, entity_domain);

 if(ent_ref == entity_undefined)
    ent_ref = make_entity(ref_full_name,
			  make_type(is_type_variable,
				    make_variable(make_basic_int(4 /* UU */),
						  NIL, NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value(is_value_unknown, UU));

 ref = make_reference(ent_ref, NIL);

/* Initialization of global variables */
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void save_int(int i): The parser has found an integer as a part of a
 * lisp expression. We save it in our global variable "lin_exp_l".
 *
 * If "lin_exp_l" is empty, then this integer becomes the current expression.
 * If not, it becomes an argument of the first lisp expression of "lin_exp_l".
 */
void save_int(i)
int i;
{
 extern list lin_exp_l;
 extern expression crt_exp;
 expression aux_exp;

 aux_exp = int_to_expression(i);

 if(lin_exp_l == NIL)
    crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
				             CONS(EXPRESSION, aux_exp, NIL));
   }
}


/*============================================================================*/
/* void save_id(string s): The parser has found a variable as a part of a
 * lisp expression. We save it in our global variable "lin_exp_l".
 *
 * If "lin_exp_l" is empty, then this variable becomes the current expression.
 * If not, it becomes an argument of the first lisp expression of "lin_exp_l".
 */
void save_id(s)
string s;
{
 extern list lin_exp_l;
 extern expression crt_exp;
 expression aux_exp;

 aux_exp = make_id_expression(s);

 if(lin_exp_l == NIL)
    crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
				             CONS(EXPRESSION, aux_exp, NIL));
   }
}


/*============================================================================*/
/* void init_op_name(string op_name): this function initializes the global
 * variable "crt_op_name". It gives the current operation that the parser is
 * dealing with.
 */
void init_op_name(op_name)
string op_name;
{
 extern string crt_op_name;

 crt_op_name = op_name;
}


/*============================================================================*/
/* void init_op_exp(string op_name): initializes a new lisp expression with
 * the operation "op_name". This expression is put at the beginning of
 * "lin_exp_l", it is the expression the parser is currently reading.
 *
 * If "op_name" is the string "0" then the operator used is "crt_op_name", else
 * the operator name is contained in "op_name".
 */
void init_op_exp(op_name)
string op_name;
{
 extern list lin_exp_l;

 lisp_expression new_le;

 if(strncmp(op_name, "0", 1) == 0)
    new_le = make_lisp_expression(crt_op_name, NIL);
 else
    new_le = make_lisp_expression(op_name, NIL);

 lin_exp_l = CONS(LISP_EXPRESSION, new_le, lin_exp_l);
}


/*============================================================================*/
/* void save_exp(): the parser has completed the reading of one lisp
 * expression, this is the first lisp expression of "lin_exp_l". We extract it
 * from this list and translate it into a Pips expression. If there is no other
 * lisp expression in "lin_exp_l", then this expression becomes the current
 * expression, else it becomes an argument of the next lisp expression which is
 * now the first object of "lin_exp_l".
 */
void save_exp()
{
 expression aux_exp;
 lisp_expression aux_le;

 aux_le = LISP_EXPRESSION(CAR(lin_exp_l));
 aux_exp = lisp_exp_to_ri_exp(aux_le);

 lin_exp_l = CDR(lin_exp_l);

 if(lin_exp_l == NIL)
   crt_exp = aux_exp;
 else
   {
    lisp_expression crt_le = LISP_EXPRESSION(CAR(lin_exp_l));
    lisp_expression_args(crt_le) = gen_nconc(lisp_expression_args(crt_le),
				             CONS(EXPRESSION, aux_exp, NIL));
   }
}


/*============================================================================*/
/* void new_df_ref_ind(char *s_ind): the parser has read a new indice of
 * the current reference. We put it in the list of indices "ref_inds".
 */
void new_df_ref_ind(string s_ind __attribute__ ((unused)))
{
 ref_inds = gen_nconc(ref_inds, CONS(EXPRESSION, crt_exp, NIL));

 /* Initialisation of the global variables used for the reference indices
  * parsing.
  */
 lin_exp_l = NIL;
}


/*============================================================================*/
/* void finish_new_df_ref(): the parser has completed the reading of the
 * current reference with all its indices. We update our current reference and
 * update all nodes contained in "crt_node_l".
 */
void finish_new_df_ref()
{
 list aux_l;

 reference_indices(ref) = ref_inds;

 for(aux_l = crt_node_l; aux_l != NIL; aux_l = CDR(aux_l))
   {
    dataflow df;

    df = first_df_of_succ(first_succ_of_vertex(VERTEX(CAR(aux_l))));
    dataflow_reference(df) = ref;
   }
}


/*============================================================================*/
/* void new_df_sink_ins(char *s_ins): the parser has read the name of the
 * sink statement. With this name we get its number. We update the global
 * list of statements and the list "crt_node_l". At this time, all the
 * informations needed for the nodes of "crt_node_l" are present, we then
 * concatenate these nodes into the list of vertices of the graph.
 */
void new_df_sink_ins(s_ins)
char * s_ins;
{
 extern list stmt_list;
 extern predicate exec_dom;

 list aux_l;

 /* In PAF, an instruction name is a string "ins_#", where "#" is the number
  * associated with the instruction. We get this number.
  */
 sink_stmt = atoi(strdup(s_ins + INS_NAME_LENGTH));

 if(find_stmt_with_num(sink_stmt) == statement_undefined)
    stmt_list = CONS(STATEMENT,
		     make_statement(entity_undefined, 1, sink_stmt,
				    strdup(s_ins), instruction_undefined,
				NIL, // No local declarations
				NULL, // null or empty string...
				empty_extensions (), make_synchronization_none()),
		     stmt_list);

 for(aux_l = crt_node_l; aux_l != NIL; aux_l = CDR(aux_l))
   {
    successor succ = first_succ_of_vertex(VERTEX(CAR(aux_l)));
    /* FI: removed because of problems. paf_ri.h and/or paf_util.h  or something else
       should be included and might cause conflict between newgen
       data structures */
    successor_vertex(succ) = vertex_undefined;
    successor_vertex(succ) = make_vertex(make_dfg_vertex_label(sink_stmt,
							       exec_dom,
							       sccflags_undefined),
					 NIL);
   }

 graph_vertices(dfg) = gen_nconc(graph_vertices(dfg), crt_node_l);
}


/*============================================================================*/
/* void init_new_do_loop(char *s_loop): initializes the parsing of the paf
 * file ".do". The parser has read the name of a loop. With this name we create
 * a new loop which becomes the current loop. The name of the loop is put in
 * the loop label.
 */
void init_new_do_loop(s_loop)
char * s_loop;
{
 extern loop crt_loop;
 entity loop_ent;
 string loop_full_name;

 loop_full_name = strdup(concatenate(DFG_MODULE_NAME,
                                     MODULE_SEP_STRING,
                                     s_loop, (char *) NULL));

 loop_ent = gen_find_tabulated(loop_full_name, entity_domain);

 if(loop_ent == entity_undefined)
    loop_ent = make_entity(loop_full_name,
                           make_type(is_type_statement,UU),
                           make_storage(is_storage_ram, ram_undefined),
                           make_value(is_value_unknown, UU));

 crt_loop = make_loop(entity_undefined,
                          make_range(expression_undefined,
                                     expression_undefined,
                                     expression_undefined),
                          statement_undefined,
                          loop_ent, execution_undefined, NIL);
}


/*============================================================================*/
/* void init_loop_ctrl(char *s_ind): initializes the parsing of the control
 * of a loop. The name that was parsed is the name of the loop index. Then we
 * create the corresponding entity and update our current loop.
 */
void init_loop_ctrl(s_ind)
char * s_ind;
{
 string index_full_name;
 entity var_ind;

 index_full_name = strdup(concatenate(DFG_MODULE_NAME, MODULE_SEP_STRING,
				      s_ind, (char *) NULL));

 var_ind = gen_find_tabulated(index_full_name, entity_domain);

 if(var_ind == entity_undefined)
    var_ind = make_entity(index_full_name,
			  make_type(is_type_variable,
				    make_variable(make_basic_int(4/*UU*/),
						  NIL, NIL)),
			  make_storage(is_storage_ram, ram_undefined),
			  make_value(is_value_unknown, UU));

 loop_index(crt_loop) = var_ind;

 lin_exp_l = NIL;
}


/*============================================================================*/
/* void lbound_exp(): The parser has read the lower bound expression of the
 * current loop. This expression is contained in "crt_exp". We update our
 * current loop.
 */
void lbound_exp()
{
 range_lower(loop_range(crt_loop)) = crt_exp;

 lin_exp_l = NIL;
}


/*============================================================================*/
/* void step_exp(): The parser has read the step expression of the current
 * loop. This expression is contained in "crt_exp". We update our current loop.
 */
void step_exp()
{
 range_increment(loop_range(crt_loop)) = crt_exp;

 lin_exp_l = NIL;
}


/*============================================================================*/
/* void ubound_exp(): The parser has read the upper bound expression of the
 * current loop. This expression is contained in "crt_exp". We update our
 * current loop.
 */
void ubound_exp()
{
 range_upper(loop_range(crt_loop)) = crt_exp;

 lin_exp_l = NIL;
}


/*============================================================================*/
/* void finish_new_do_loop(): This function update the global list of loops
 * with the current loop.
 */
void finish_new_do_loop()
{
 extern list loop_list;

 loop_list = CONS(LOOP, crt_loop, loop_list);
}


/*============================================================================*/
/* void init_new_gd_ins(char *s_ins): initializes the parsing of the paf
 * file ".gd". The parser has read the name of a statement. We get the number
 * contained in this name and put in our current statement "crt_stmt".
 *
 * "crt_el" is the list of the current englobing loops, i.e. the englobing
 * loops of the current statement. It is initializes to NIL.
 */
void init_new_gd_ins(s_ins)
char * s_ins;
{
 extern list crt_el;
 extern int crt_stmt;

 crt_stmt = atoi(strdup(s_ins + INS_NAME_LENGTH));
 crt_el = NIL;
}


/*============================================================================*/
/* static loop find_loop_with_name(string s): returns the loop that has a label
 * name equal to the name given in argument ("s"). This function uses the
 * global list of loops "loop_list".
 */
static loop find_loop_with_name(s)
string s;
{
 extern list loop_list;

 list aux_l = loop_list;

 for(; aux_l != NIL; aux_l = CDR(aux_l))
   {
    loop aux_loop = LOOP(CAR(aux_l));
    const char* loop_name = entity_local_name(loop_label(aux_loop));
    if(strcmp(loop_name, s) == 0)
       return(aux_loop);
   }
 return(loop_undefined);
}


/*============================================================================*/
/* void new_eng_loop(char *s_loop): the parser has found a new englobing
 * loop. If it does not exist yet (we call find_loop_with_name()) we create it.
 * We update "crt_el" with this loop (at the end): we want to construct an
 * ordered list from the most external list to the innermost loop, and the
 * parsing gets the loops in this order.
 *
 */
void new_eng_loop(s_loop)
char * s_loop;
{
 extern list crt_el;

 loop aux_loop;

 aux_loop = find_loop_with_name(s_loop);
 if(aux_loop == loop_undefined)
   {
    entity loop_ent;
    string loop_full_name;

    loop_full_name = strdup(concatenate(DFG_MODULE_NAME,
					MODULE_SEP_STRING,
					s_loop, (char *) NULL));

    loop_ent = gen_find_tabulated(loop_full_name, entity_domain);

    if(loop_ent == entity_undefined)
       loop_ent = make_entity(loop_full_name,
			      make_type(is_type_statement,UU),
			      make_storage(is_storage_ram, ram_undefined),
			      make_value(is_value_unknown, UU));

    aux_loop =  make_loop(entity_undefined,
			  make_range(expression_undefined,
				     expression_undefined,
				     expression_undefined),
			  statement_undefined,
			  loop_ent, execution_undefined, NIL);
   }
 crt_el = gen_nconc(crt_el, CONS(LOOP, aux_loop, NIL));
}


/*============================================================================*/
/* void finish_new_gd_ins(): completes the parsing of the DFG. We copy the
 * list of englobing loops in order to store it in the global static
 * control map (it contains the static_controls associated to
 * the statements).  */
void finish_new_gd_ins()
{
    extern list crt_el, param_l;
    extern int crt_stmt;

    list new_el;

    /* We compute the execution domain (for the dfg) for all sink instructions
     * named "crt_stmt".
     comp_exec_domain(dfg, crt_stmt, crt_el);
     */

    /* Make a copy to store in the hash table "STS" */
    new_el = NIL;
    MAPL(cl, {
	/* loop l = EFFECT(CAR(cl)); */
	loop l = LOOP(CAR(cl));
	loop nl = loop_dup(l);
	new_el = gen_nconc(new_el, CONS(LOOP, nl, NIL));
    }, crt_el);

    /* should be a long int for crt_stmt */
    hash_put(STS, (void *) ((long)crt_stmt),
	     (void *) make_static_control(true, param_l, new_el, NIL));
}

