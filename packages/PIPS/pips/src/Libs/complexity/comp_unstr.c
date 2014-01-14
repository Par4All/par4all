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
/* comp_unstr.c */
/* evaluate the complexity of unstructured graphs of statements */

#include <stdio.h>
#include <math.h>

#include "linear.h"

#include "genC.h"

#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"  /* used by get_bool_property   */
#include "misc.h"
#include "matrice.h"
#include "complexity.h"

/* Added by AP, March 15th 93: allows the simulation of a two-dimensional array
 * with a mono-dimensional memory allocation.
 */
#define FA(i,j) fa[(i)*n_controls + (j)]

extern hash_table hash_callee_to_complexity;
extern hash_table hash_complexity_parameters;

/* 6th element of instruction */
/* complexity unstructured_to_complexity(unstructured unstr;
 *                                       transformer precond;
 *                                       list effects_list;
 *
 * Return the complexity of the unstructured graph unstr
 * whose nodes are controls.
 *
 * First runs through the graph to compute nodes' complexities,
 * the total number of nodes, and give a number to each node.
 * This is done in "controls_to_hash_table()".
 *
 * Then runs again through it to fill the probability matrix [Pij]
 * ( Pij = probability to go to node j when you're in node i)
 * This is done in "build_probability_matrix()"
 *
 * Then computes A = I-P, then inv_A = 1/A with the matrix package.
 *
 * MAX_CONTROLS_IN_UNSTRUCTURED = 100
 */
complexity unstructured_to_complexity(unstr, precond, effects_list)
unstructured unstr;
transformer precond;
list effects_list;
{
    complexity comp = make_zero_complexity(); 
    hash_table hash_control_to_complexity = hash_table_make(hash_pointer, 
					    MAX_CONTROLS_IN_UNSTRUCTURED);
    int i, j, n_controls = 0;
    control control_array[MAX_CONTROLS_IN_UNSTRUCTURED];
    matrice P, A, B;

    trace_on("unstructured");

    controls_to_hash_table(unstructured_control(unstr),
			   &n_controls,
			   control_array,
			   hash_control_to_complexity,
			   precond,
			   effects_list);
    
     
    P = average_probability_matrix(unstr, n_controls, control_array);
    A = matrice_new(n_controls, n_controls);
    B = matrice_new(n_controls, n_controls);

    /* A is identity matrix I */
    matrice_identite(A, n_controls, 0);   
    /* B = A - P =  I - P */
    matrice_substract(B, A, P, n_controls, n_controls);  

    if (get_debug_level() >= 5) {
	fprintf(stderr, "I - P =");
	matrice_fprint(stderr, B, n_controls, n_controls);
    }

    /* matrice_general_inversion(B, A, n_controls);  */    
    /* A = 1/(B) */
    /* This region will call C routines. LZ 20/10/92  */

    /* 
       the "global" complexity is:
       comp = a11 * C1 + a12 * C2 + ... + a1n * Cn
       where Ci = complexity of control #i
     */

    if ( n_controls < MAX_CONTROLS_IN_UNSTRUCTURED ) {

	/* Modif by AP, March 15th 93: old version had non constant int in dim decl
	float fa[n_controls][n_controls];
	int indx[n_controls];
	int d;
	*/
	float *fa = (float *) malloc(n_controls*n_controls*sizeof(float));
	int *indx = (int *) malloc(sizeof(int) * n_controls);
	int d;

	for (i=1; i<=n_controls; i++) {
	    for (j=1; j<=n_controls; j++ ) {
		Value n = ACCESS(B, n_controls, i, j),
		      de = DENOMINATOR(B);
		float f1 = VALUE_TO_FLOAT(n),
		      f2 = VALUE_TO_FLOAT(de);
		FA(i-1,j-1) = f1/f2;
	    }
	}

	if ( get_debug_level() >= 5 ) {
	    fprintf(stderr, "Before float matrice inversion\n\n");
	}

	if ( get_debug_level() >= 9 ) {
	    fprintf(stderr, "(I - P) =\n");
	    for (i=0;i<n_controls;i++) {
		for (j=0;j<n_controls;j++)
		   fprintf(stderr, "%4.2f ",FA(i,j) );
		fprintf(stderr, "\n");
	    }
	}

	float_matrice_inversion(fa, n_controls, indx, &d);

	if ( get_debug_level() >= 5 ) {
	    fprintf(stderr, "After  float matrice inversion\n\n");
	}

	if ( get_debug_level() >= 9 ) {
	    fprintf(stderr, "(I - P)^(-1) =\n");
	    for (i=0;i<n_controls;i++) {
		for (j=0;j<n_controls;j++)
		   fprintf(stderr, "%4.2f ",FA(i,j) );
		fprintf(stderr, "\n");
	    }
	}

	for (i=1; i<=n_controls; i++) {
	    control conti = control_array[i];
	    complexity compi = (complexity) 
		hash_get(hash_control_to_complexity, (char *) conti);
	    float f = FA(0,i-1);

	    if ( get_debug_level() >= 5 ) {
		fprintf(stderr, "control $%p:f=%f, compl.=", conti, f);
		complexity_fprint(stderr, compi, true, false);
	    }

	    complexity_scalar_mult(&compi, f);
	    complexity_add(&comp, compi);
	}
    }
    else
	pips_internal_error("Too large to compute");

    if (get_debug_level() >= 5) {
	fprintf(stderr, "cumulated complexity=");
	complexity_fprint(stderr, comp, true, false);
    }

    matrice_free(B);
    matrice_free(A);
    matrice_free(P);

    hash_table_free(hash_control_to_complexity);

    complexity_check_and_warn("unstructured_to_complexity", comp);    

    trace_off();
    return(comp);
}


/* Returns the hash table hash_control_to_complexity filled in
 * with the complexities of the successors of control cont.
 * each control of the graph is also stored in the array
 * control_array (beginning at 1).
 * also returns the total number of controls in the unstructured
 */
void controls_to_hash_table(cont, pn_controls, control_array,
			    hash_control_to_complexity, precond, effects_list)
control cont;
int *pn_controls;
control control_array[];
hash_table hash_control_to_complexity;
transformer precond;
list effects_list;
{
    statement s = control_statement(cont);
    complexity comp;

    comp = statement_to_complexity(s, precond, effects_list);
    hash_put(hash_control_to_complexity, (char *) cont, (char *) comp);
    control_array[++(*pn_controls)] = cont;
    complexity_check_and_warn("control_to_complexity", comp);

    if (get_debug_level() >= 5) {
	fprintf(stderr, "this control($%p) has:", cont);
	complexity_fprint(stderr, comp, true, true);
	MAPL(pc, {
	    fprintf(stderr, "successor: $%p\n", CONTROL(CAR(pc)));
	    }, control_successors(cont));
	if ( control_successors(cont) == NIL )
	    fprintf(stderr, "NO successor at all!\n");
	fprintf(stderr, ". . . . . . . . . . . . . . . . . . . . . .\n");
    }

    MAPL(pc, {
	control c = CONTROL(CAR(pc));
	if ( hash_get(hash_control_to_complexity, (char *) c)
	    ==HASH_UNDEFINED_VALUE )
	    controls_to_hash_table(c, pn_controls, control_array,
				   hash_control_to_complexity, precond,
				   effects_list);
    }, control_successors(cont));
}

/* return the number i, that is i'th element of the control array
 * Note that i begins from 1 instead of 0 
 */
int control_element_position_in_control_array(cont, control_array, n_controls)
control cont;
control control_array[];
int n_controls;
{
    int i;
    
    for (i=1; i<=n_controls; i++)
	if (cont == control_array[i]) 
	    return (i);
    pips_internal_error("this control isn't in control_array[]!");

    /* In order to satisfy compiler. LZ 3 Feb. 93 */
    return (i);
}


matrice average_probability_matrix(unstr, n_controls, control_array)
unstructured unstr;
int n_controls;
control control_array[];
{
    control cont = unstructured_control(unstr);
    bool already_examined[MAX_CONTROLS_IN_UNSTRUCTURED];
    int i, j , n_succs, max_n_succs = 0;
    matrice P = matrice_new(n_controls, n_controls);

    if ( n_controls > MAX_CONTROLS_IN_UNSTRUCTURED ) {
	pips_internal_error("control number is larger than %d", 
		   MAX_CONTROLS_IN_UNSTRUCTURED );
    }

    matrice_nulle(P, n_controls, n_controls);

    /* initilize the already_examined to false */
    for (i=1; i<=n_controls; i++) 
	already_examined[i] = false;

    /* make in P the matrix "is_successor_of" */
    node_successors_to_matrix(cont, P, n_controls,
			      control_array, already_examined);

    /* we'll attribute equitable probabilities to the n succ. of a node */
    for (i=1; i<=n_controls; i++) {
	n_succs = 0;
	for (j=1; j<=n_controls; j++)
	{ 
	    Value a = ACCESS(P, n_controls, i, j);
	    n_succs += VALUE_TO_INT(a);
	}
	if (n_succs > max_n_succs) 
	    max_n_succs = n_succs;
    }

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "n_controls=%d, and max_n_succs=%d\n",
		n_controls,max_n_succs);
    }

    /* computes probabilities (0<=p<1) out of the matrix "is_successor_of" */

    
    DENOMINATOR(P) = int_to_value(factorielle(max_n_succs));

    for (i=1; i<=n_controls; i++) {
	n_succs = 0;
	for (j=1; j<=n_controls; j++) {
	    Value a = ACCESS(P, n_controls, i, j);
	    n_succs += VALUE_TO_INT(a);
	}
	if (n_succs>0)
	    for (j=1; j<=n_controls; j++)
	    {
		Value x = value_div(DENOMINATOR(P),int_to_value(n_succs));
		value_product(ACCESS(P, n_controls, i, j),x) ; 
	    }
    }

    matrice_normalize(P, n_controls, n_controls);

    if (get_debug_level() > 0) {
	fprintf(stderr, "n_controls is %d\n", n_controls);
	fprintf(stderr, "average_probability_matrix:  P =\n");
	matrice_fprint(stderr, P, n_controls, n_controls);
    }

    return (P);
}

/* 
 * On return, Pij = 1  <=>  there's an edge from control #i to #j 
 * It means that every succeccor has the same possibility to be reached.
 *
 * Modification:
 *  - put already_examined[i] = true out of MAPL.
 *    If control i has several successors, there is no need to set it several
 *    times in MAPL. LZ 13/04/92
 */
void node_successors_to_matrix(cont, P, n_controls,
			       control_array, already_examined)
control cont;
matrice P;
int n_controls;
control control_array[];
bool already_examined[];
{
    int i = control_element_position_in_control_array(cont, control_array, n_controls);

    already_examined[i] = true;

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "CONTROL ($%p)  CONTROL_NUMBER %d\n", cont, i);
    }

    MAPL(pc, {
	control succ = CONTROL(CAR(pc));
	int j = control_element_position_in_control_array(succ, control_array, 
							  n_controls);

	if ( get_debug_level() >= 5 ) {
	    fprintf(stderr,"Control ($%p) %d  -->  Control ($%p) %d\n",
		    cont, i, succ, j);
	}

	/* Here, we give equal possibility , 1 for each one */
	ACCESS(P, n_controls, i, j) = VALUE_ONE;
	if (!already_examined[j])
	    node_successors_to_matrix(succ, P, n_controls,
				      control_array, already_examined);
	else {
	    if ( get_debug_level() >= 5 ) {
		fprintf(stderr, "Control Number %d already examined!\n",j);
	    }
	}
    }, control_successors(cont));
}
