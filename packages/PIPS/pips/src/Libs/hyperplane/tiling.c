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
/* package tiling
 *
 * 1. Why?
 *    - memory hierarchy (registers, caches L1/L2/L3, memory, virtual memory, out-of-core,...)
 *    - granularity (amortization of fix costs: synchronization, communication, control,...)
 * 2. Legality
 *    - TO BE IMPLEMENTED
 * 3. Selection
 *    - directions (e.g. communication minimization): Darte/Robert, Hoegsted
 *    - ratios (e.g. critical path minimization)
 *    - volume (fix cost amortization) under memory constraints
 *    - mapping on finite hardware (Robert/...)
 *    restriction to orthogonal tilings: ratios, volume, mapping (Andonov/Rajopadhye)
 * 4. Code Generation (Xue?)
 *    - control and memory addressing overheads
 * 5. Hierarchical Tiling (Ferrante/Carter,...)
 * 6. Data vs Control Tiling
 * 7. Extensions
 *    - perfectly nested loops (IMPLEMENTED)
 *    - non perfectly nested loops (e.g. matrix multiply)
 *    - general nested loops (e.g. ADI)
 *    - sequence of loop nests (signal processing, Thomson-CSF)
 *    - ...
 * See: http://www.irisa.fr/api/Rajopadhye/tiling
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "misc.h"
#include "text.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "matrix.h"

#include "sparse_sc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "conversion.h"

#include "transformations.h"
#include "properties.h"

#include "hyperplane.h"

/* Create a new entity for tile index. Because of the module name, it is easier to postfix
 * by "_t" than to prefix by "t_".
 */
static entity 
make_tile_index_entity(entity old_index)
{
  return make_new_index_entity(old_index, "_t");
}

bool
static_partitioning_matrix(matrice P, int n, const char* serialized_matrix)
{
    pips_assert("interactive_partitioning_matrix", n>=1);
    bool status = false;
    if( serialized_matrix && !string_undefined_p(serialized_matrix) && !empty_string_p(serialized_matrix))
    {
        int row,col;
        string ctxt0,ctxt1;
        string saved_ptr,buffer,elem;
        saved_ptr= buffer = strdup(serialized_matrix);
        string line = strtok_r(buffer,",",&ctxt0);
        DENOMINATOR(P) = VALUE_ONE;
        for(row=1;row<=n;row++){
            elem = strtok_r(line," ",&ctxt1);

            for(col=1;col<=n;col++)
            {
                if(!elem) elem = col==row ? "1" : "0";
                ACCESS(P, n, row, col)=atoi(elem);
                elem = strtok_r(NULL," ",&ctxt1);
            }
            line = strtok_r(NULL,",",&ctxt0);
        }
        status= ( line == NULL ) && (elem == NULL );
        free(saved_ptr);
    }
    return status;
}

/* Query the user for a partitioning matrix P
 */

bool
interactive_partitioning_matrix(matrice P, int n)
{
    int n_read;
    string resp = string_undefined;
    string cn = string_undefined;
    bool return_status = false;
    int row;
    int col;

    DENOMINATOR(P) = VALUE_ONE;
    /* Query the user for P's components */
    pips_assert("interactive_partitioning_matrix", n>=1);
    debug(8, "interactive_partitioning_matrix", "Reading P\n");

    for(row=1; row<=n; row++) {
	resp = user_request("Partitioning matrix (%dx%d)?\n"
			    "(give all its integer coordinates on one line of %d per row): ",
			    n, n, n);
	if (resp[0] == '\0') {
	    user_log("Tiling loop transformation has been cancelled.\n");
	    return_status = false;
	}
	else {    
	    cn = strtok(resp, " \t");

	    return_status = true;
	    for( col = 1; col<=n; col++) {
		if(cn==NULL) {
		    user_log("Too few coordinates. "
			     "Tiling loop transformation has been cancelled.\n");
		    return_status = false;
		    break;
		}
		n_read = sscanf(cn," " VALUE_FMT, &ACCESS(P, n, row, col));
		if(n_read!=1) {
		    user_log("Too few coordinates. "
			     "Hyperplane loop transformation has been cancelled.\n");
		    return_status = false;
		    break;
		}
		cn = strtok(NULL, " \t");
	    }
	}

	if(cn!=NULL) {
	    user_log("Too many coordinates. "
		     "Tiling loop transformation has been cancelled.\n");
	    return_status = false;
	}
    }

    ifdebug(8) {
	if(return_status) {
	    pips_debug(8, "Partitioning matrix:\n");
	    matrice_fprint(stderr, P, n, n);
	    (void) fprintf(stderr,"\n");
	    pips_debug(8, "End\n");
	}
	else {
	    pips_debug(8, "Ends with failure\n");
	}
    }

    return return_status;
}


/* Generate the tile membership constraints between a tile coordinates and
 an iteration coordinate
 */
static Psysteme
tile_membership_constraints(Pbase initial_basis,
			    Pbase tile_basis,
			    matrice HT,
			    Pvecteur tiling_offset)
{
    Psysteme mc = sc_new();
    int dim = base_dimension(initial_basis);
    int row;
    int col;
    Value k = DENOMINATOR(HT);
    Value up = k - VALUE_ONE;
    Pbase civ = BASE_UNDEFINED;
    Pbase ctv = BASE_UNDEFINED;

    ifdebug(8) {
	debug(8, "tile_membership_constraints", "Begin with Matrix HT:\n");
	matrice_fprint(stderr, HT, dim, dim);
    }

    pips_assert("The two bases have the same dimension", dim == base_dimension(tile_basis));

    for(row = 1; row <= dim; row++) {
	Pvecteur upper = VECTEUR_NUL;
	Pvecteur lower = VECTEUR_NUL;
	Pcontrainte cupper = CONTRAINTE_UNDEFINED;
	Pcontrainte clower = CONTRAINTE_UNDEFINED;

	for(col = 1, civ = initial_basis, ctv = tile_basis;
	    col <= dim;
	    col++, civ = vecteur_succ(civ), ctv = vecteur_succ(ctv)) {
	    if(ACCESS(HT, dim, row, col)!=VALUE_ZERO) {
		Value coeff = ACCESS(HT, dim, row, col);
		Value offset = vect_coeff(vecteur_var(civ), tiling_offset);

		vect_add_elem(&upper, vecteur_var(civ), coeff);
		vect_add_elem(&upper, TCST, value_uminus(offset*coeff));
	    }
	    if(col==row) {
		vect_add_elem(&upper, vecteur_var(ctv), value_uminus(k));
	    }
	}
	lower = vect_dup(upper);
	vect_chg_sgn(lower);
	vect_add_elem(&upper, TCST, value_uminus(up));
	cupper = contrainte_make(upper);
	clower = contrainte_make(lower);
	sc_add_inegalite(mc, cupper);
	sc_add_inegalite(mc, clower);
    }

    sc_creer_base(mc);

    ifdebug(8) {
	pips_debug(8, "End with constraint system mc:\n");
	sc_fprint(stderr, mc, (get_variable_name_t)entity_local_name);
    }

    return mc;
}

/* Find the origin of the iteration domain. Use 0 as default coordinate */

Pvecteur
loop_nest_to_offset(list lls)
{
    Pvecteur origin = VECTEUR_NUL;
    list cs = list_undefined;

    for (cs = lls; cs != NIL; cs = CDR(cs)){
	loop l = instruction_loop(statement_instruction(STATEMENT(CAR(cs))));
	entity ind = loop_index(l);
	range r = loop_range(l);
	expression lower = range_lower(r);
	intptr_t val;

	if(expression_integer_value(lower, &val)) {
	    vect_chg_coeff(&origin, (Variable) ind, (Value) val);
	}
    }

    return origin;
}

/* Generate tiled code for a loop nest, PPoPP'91, p. 46, Figure 15.
 *
 * The row-echelon algorithm is called from new_loop_bound().
 */

statement tiling( list lls)
{
    Psysteme sci;			/* iteration domain */
    Psysteme sc_tile_scan;
    Psysteme sc_tile;
    Psysteme mc = SC_UNDEFINED; /* Tile membership constraints */
    Psysteme sc_B_prime = SC_UNDEFINED;
    Psysteme sc_B_second = SC_UNDEFINED;
    Pbase initial_basis = NULL;
    Pbase tile_basis = NULL;
    Pbase reverse_tile_basis = NULL;
    /* Pbase local_basis = NULL; */
    Pbase new_basis = NULL;
    matrice P; /* Partitioning matrix */
    matrice HT; /* Transposed matrix of the inverse of P */
    matrice G; /* Change of basis in the tile space to use vector 1 as hyperplane direction */
    int n;				/* number of indices, i.e. loop nest depth */
    Value *h;
    statement s_lhyp;
    Pvecteur *pvg;
    Pbase pb;
    expression lower, upper;
    int col;
    Pvecteur to = VECTEUR_NUL; /* Tiling offset: 0 by default */

    debug_on("TILING_DEBUG_LEVEL");

    pips_debug(8, "Begin with iteration domain:\n");

    /* make the constraint system for the iteration space and find a good
       origin for the tiling */

    sci = loop_iteration_domaine_to_sc(lls, &initial_basis);
    n = base_dimension(initial_basis);
    to = loop_nest_to_offset(lls);
    ifdebug(8) {
	sc_fprint(stderr, sci, (get_variable_name_t)entity_local_name);
	pips_debug(8,"And with origin:\n");
	vect_fprint(stderr, to, (get_variable_name_t)entity_local_name);
    }

    /* computation of the partitioning matrix P and its inverse HT */

    P = matrice_new(n, n);
    HT = matrice_new(n, n);
    const char* smatrix = get_string_property("LOOP_TILING_MATRIX");

    if(
            !static_partitioning_matrix(P,n,smatrix) &&
            !interactive_partitioning_matrix(P, n)
      ) {
        pips_user_error("A proper partitioning matrix was not provided\n");
    }

    ifdebug(8) {
	pips_debug(8,"Partitioning matrix P:");
	matrice_fprint(stderr, P, n, n);
	(void) fprintf(stderr,"\n");
    }

    matrice_general_inversion(P, HT, n);

    ifdebug(8) {
	pips_debug(8,"Inverse partitioning matrix HT:");
	matrice_fprint(stderr, HT, n, n);
	(void) fprintf(stderr,"\n");
    }

    /* Compute B': each iteration i in the iteration space is linked to its tile s */

    derive_new_basis(initial_basis, &tile_basis, make_tile_index_entity);
    mc = tile_membership_constraints(initial_basis, tile_basis, HT, to);
    mc = sc_normalize(mc);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile membership constraints:\n");
	sc_fprint(stderr, mc, (get_variable_name_t)entity_local_name);
    }
    /* mc and SC_B_prime are aliased after this call */
    sc_B_prime = sc_append(mc, sci);
    ifdebug(8) {
	(void) fprintf(stderr,"sc_B_prime after call to sc_append (is the basis ok?):\n");
	sc_fprint(stderr, sc_B_prime, (get_variable_name_t)entity_local_name);
    }
    mc = SC_UNDEFINED;
    /* Save a copy to compute B" later */
    sc_B_second = sc_dup(sc_B_prime);

    /* Get constraints on tile coordinates */

    sc_projection_along_variables_ofl_ctrl(&sc_B_prime, initial_basis, OFL_CTRL);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile domain:\n");
	sc_fprint(stderr, sc_B_prime, (get_variable_name_t)entity_local_name);
    }

    /* Build the constraint system to scan the set of tiles */
    sc_tile_scan = new_loop_bound(sc_B_prime, tile_basis);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile domain in echelon format:\n");
	sc_fprint(stderr, sc_tile_scan, (get_variable_name_t)entity_local_name);
    }

    /* CA: Build the new basis (tile_basis+initial_basis)*/
    /* base It, Jt, I, J  pour notre exemple */
    new_basis = vect_add(vect_dup(initial_basis),vect_dup(tile_basis));
    ifdebug(8) {
	(void) fprintf(stderr,"new_basis\n");
	vect_fprint(stderr, new_basis, (get_variable_name_t)entity_local_name);
    }

    /* Build the constraint system sc_tile to scan one tile (BS IN
       PPoPP'91 paper) */
    ifdebug(8) {
	(void) fprintf(stderr,"sc_B_second:\n");
	sc_fprint(stderr, sc_B_second, (get_variable_name_t)entity_local_name);
    }
    sc_tile = new_loop_bound(sc_B_second, new_basis);
    ifdebug(8) {
	(void) fprintf(stderr,"Iteration domain for one tile:\n");
	sc_fprint(stderr, sc_tile, (get_variable_name_t)entity_local_name);
    }


    /* computation of the hyperplane tile direction: let's use the
       default 1 vector */
    h = (Value*)(malloc(n*sizeof(Value)));
    for(col=0; col<n; col++) {
	h[col] = VALUE_ONE;
    }
    /* computation of the tile scanning base G: right now, let's
     * assume it's Id.  This is OK to tile parallel loops... or to
     * scan tiles sequentially on a monoprocessor.
     */
    G = matrice_new(n,n);
    scanning_base_hyperplane(h, n, G);
    matrice_identite(G, n, 0);
    ifdebug(8) {
	(void) fprintf(stderr,"The tile scanning base G is:");
	matrice_fprint(stderr, G, n, n);
    }

    /* generation of code for scanning one tile */

    /* Compute the local coordinate changes: there should be none for the time being
     * because we keep the initial basis to scan iterations within one tile, i.e
     * G must be the identity matrix
     */
    pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
    scanning_base_to_vect(G, n, initial_basis, pvg);

    /* generation of code to scan one tile and update of loop body using pvg */

    s_lhyp = code_generation(lls, pvg, initial_basis,
			     new_basis, sc_tile, false);

    /* generation of code for scanning all tiles */

    reverse_tile_basis = base_reversal(tile_basis);
    for (pb = reverse_tile_basis; pb!=NULL; pb=pb->succ) {
	loop tl = loop_undefined;

	make_bound_expression(pb->var, tile_basis, sc_tile_scan, &lower, &upper);
	tl = make_loop((entity) vecteur_var(pb),
		       make_range(copy_expression(lower), copy_expression(upper),
				  int_to_expression(1)),
		       s_lhyp,
		       entity_empty_label(),
		       make_execution(is_execution_sequential, UU),
		       NIL);
	s_lhyp = instruction_to_statement(make_instruction(is_instruction_loop, tl));
    }

    pips_debug(8,"End\n");

    debug_off();

    return(s_lhyp);
}

bool
loop_tiling(const char* module_name)
{
    bool return_status = false;

    return_status = interactive_loop_transformation(module_name,  (statement (*)(list, bool (*)(statement)))tiling);
    
    return return_status;
}
