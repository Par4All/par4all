/* package hyperplane
 *
 * $Id$
 * 
 * $Log: tiling.c,v $
 * Revision 1.3  1998/10/12 17:00:37  ancourt
 * essai ca code generation
 *
 * Revision 1.2  1998/10/12 16:25:51  ancourt
 * *** empty log message ***
 *
 * Revision 1.1  1998/10/12 10:03:33  irigoin
 * Initial revision
 *
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <strings.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
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
#include "prettyprint.h"
#include "conversion.h"

#include "transformations.h"

#include "hyperplane.h"

/* Create a new entity for tile index. Because of the module name, it is easier to postfix
 * by "_t" than to prefix by "t_".
 */
static entity 
make_tile_index_entity(entity old_index)
{
    entity new_index;
    string old_name;
    char *new_name = (char*) malloc(33);

    old_name = entity_name(old_index);
    for (sprintf(new_name, "%s%s", old_name, "_t");
         gen_find_tabulated(new_name, entity_domain)!=entity_undefined; 

         old_name = new_name) {
        sprintf(new_name, "%s%s", old_name, "_t");
	pips_assert("Variable name cannot be longer than 32 characters",
		    strlen(new_name)<33);
    }
 
   new_index = make_entity(new_name,
			   copy_type(entity_type(old_index)),
			   /* Should be AddVariableToCommon(DynamicArea) or
			      something similar! */
			   copy_storage(entity_storage(old_index)),
			   copy_value(entity_initial(old_index)));

    return(new_index);
}

/* Query the user for a partitioning matrix P
 */

bool
interactive_partitioning_matrix(matrice P, int n)
{
    int n_read;
    string resp = string_undefined;
    string cn = string_undefined;
    bool return_status = FALSE;
    int row;
    int col;

    DENOMINATOR(P) = VALUE_ONE;
    /* Query the user for P's components */
    pips_assert("interactive_partitioning_matrix", n>=1);
    debug(8, "interactive_partitioning_matrix", "Reading P\n");

    for(row=1; row<=n; row++) {
	resp = user_request("Partitioning matrix?\n"
			    "(give all its integer coordinates on one line per row): ");
	if (resp[0] == '\0') {
	    user_log("Tiling loop transformation has been cancelled.\n");
	    return_status = FALSE;
	}
	else {    
	    cn = strtok(resp, " \t");

	    return_status = TRUE;
	    for( col = 1; col<=n; col++) {
		if(cn==NULL) {
		    user_log("Too few coordinates. "
			     "Tiling loop transformation has been cancelled.\n");
		    return_status = FALSE;
		    break;
		}
		n_read = sscanf(cn," " VALUE_FMT, &ACCESS(P, n, row, col));
		if(n_read!=1) {
		    user_log("Too few coordinates. "
			     "Hyperplane loop transformation has been cancelled.\n");
		    return_status = FALSE;
		    break;
		}
		cn = strtok(NULL, " \t");
	    }
	}

	if(cn!=NULL) {
	    user_log("Too many coordinates. "
		     "Tiling loop transformation has been cancelled.\n");
	    return_status = FALSE;
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


/* Generate the tile membership constraints
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
		vect_add_elem(&upper, vecteur_var(civ), coeff);
	    }
	    if(col==row) {
		/* For the time being, I do not know what to do with the offset! */
		/* Value offset = vect_coeff(civ, tiling_offset); */
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
	debug(8, "tile_membership_constraints", "End with constraint system mc:\n");
	sc_fprint(stderr, mc, entity_local_name);
    }

    return mc;
}


/* Generate tiled code for a loop nest, PPoPP'91
 */

statement 
tiling( list lls)
{
    Psysteme sci;			/* iteration domain */
    Psysteme scn;			/* sc nouveau */
    Psysteme sc_tile_scan;
    Psysteme sc_tile;
    Psysteme sc_newbase;
    Psysteme mc = SC_UNDEFINED; /* Tile membership constraints */
    Psysteme sc_B_prime = SC_UNDEFINED;
    Psysteme sc_B_second = SC_UNDEFINED;
    Pbase initial_basis = NULL;
    Pbase tile_basis = NULL;
    Pbase local_basis = NULL;
    Pbase new_basis = NULL;
    matrice B; /* Constraints for loop bounds */
    matrice P; /* Partitioning matrix */
    matrice HT; /* Transposed matrix of the inverse of P */
    /*
    matrice S;
    matrice AG;
    matrice A;
    */
    matrice G; /* Change of basis in the tile space to use vector 1 as hyperplane direction */
    matrice PG; /* Change of basis from the hyperplaned tile space to the iteration space */
    matrice BPG; /* Constraints for the tile coordinates */
    int n;				/* number of indices, i.e. loop nest depth */
    int m ;				/* number of constraints */
    Value *h;
    statement s_lhyp;
    Pvecteur *pvg;
    Pbase pb;  
    expression lower, upper;
    Pvecteur pv1, pv2;
    loop l;
    int col;
    Pvecteur to = VECTEUR_NUL; /* Tiling offset: 0 by default */

    debug_on("TILING_DEBUG_LEVEL");

    debug(8,"tiling","Begin:\n");

    /* make the  constraint system for the iteration space */
    sci = loop_iteration_domaine_to_sc(lls, &initial_basis);
    
    /* create the constraint matrix B for the loop bounds */
    n = base_dimension(initial_basis);
    m = sci->nb_ineq;
    B = matrice_new(m,n);
    sys_matrice_index(sci, initial_basis, B, n, m);

    /* computation of the partitioning matrix P and its inverse HT */
    P = matrice_new(n, n);
    HT = matrice_new(n, n);

    if(!interactive_partitioning_matrix(P, n)) {
	pips_user_error("A proper partitioning matrix was not provided\n");
    }

    ifdebug(8) {
	debug(8,"tiling","Partitioning matrix P:");
	matrice_fprint(stderr, P, n, n);
	(void) fprintf(stderr,"\n");
    }

    matrice_general_inversion(P, HT, n);

    ifdebug(8) {
	debug(8,"tiling","Inverse partitioning matrix HT:");
	matrice_fprint(stderr, HT, n, n);
	(void) fprintf(stderr,"\n");
    }

    /* Compute B': each iteration i in the iteration space is linked to its tile s */
    derive_new_basis(initial_basis, &tile_basis, make_tile_index_entity);
    mc = tile_membership_constraints(initial_basis, tile_basis, HT, to);
    mc = sc_normalize(mc);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile membership constraints:\n");
	sc_fprint(stderr, mc, entity_local_name);
    }
    /* mc and SC_B_prime are aliased after this call */
    sc_B_prime = sc_append(mc, sci);
    ifdebug(8) {
	(void) fprintf(stderr,"sc_B_prime after call to sc_append (is the basis ok?):\n");
	sc_fprint(stderr, sc_B_prime, entity_local_name);
    }
    mc = SC_UNDEFINED;
    sc_B_second = sc_dup(sc_B_prime);

    /* Get constraints on tile coordinates */
    sc_projection_along_variables_ofl_ctrl(&sc_B_prime, initial_basis, OFL_CTRL);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile domain:\n");
	sc_fprint(stderr, sc_B_prime, entity_local_name);
    }

    /* Build code to scan the tiles */
    sc_tile_scan = new_loop_bound(sc_B_prime, tile_basis);
    ifdebug(8) {
	(void) fprintf(stderr,"Tile domain in echelon format:\n");
	sc_fprint(stderr, sc_tile_scan, entity_local_name);
    }
    /* Build the new basis (tile_basis+initial_basis)*/
    /* base It, Jt, I, J  pour notre exemple */ 
    new_basis = vect_add(vect_dup(initial_basis),vect_dup(tile_basis)
			 );
 ifdebug(8) {
	(void) fprintf(stderr,"new_basis\n");
	vect_fprint(stderr, new_basis, entity_local_name);
    }
    /* Build the code to scan any one tile */
    ifdebug(8) {
	(void) fprintf(stderr,"sc_B_second:\n");
	sc_fprint(stderr, sc_B_second, entity_local_name);
    }
    sc_tile = new_loop_bound(sc_B_second, new_basis);
    ifdebug(8) {
	(void) fprintf(stderr,"Iteration domain for one tile:\n");
	sc_fprint(stderr, sc_tile, entity_local_name);
    }


    /* computation of the hyperplane tile direction: let's use the default 1 vector */
    h = (Value*)(malloc(n*sizeof(Value)));
    for(col=0; col<n; col++) {
	h[col] = VALUE_ONE;
    }

    /* computation of the tile scanning base G: right now, let's assume it's Id.
     * This is OK to tile parallel loops
     */
    G = matrice_new(n,n); 
    scanning_base_hyperplane(h, n, G);	  
    matrice_identite(G, n, 0);
    ifdebug(8) {
	(void) fprintf(stderr,"The tile scanning base G is:");
	matrice_fprint(stderr, G, n, n);
    }
    
    /* generation of code for scanning all tiles */
    /*  generation of bounds */
    /* est completement inutile 
    for (pb = tile_basis; pb!=NULL; pb=pb->succ) {
	make_bound_expression(pb->var, tile_basis, sc_tile_scan, &lower, &upper);
    } */ 
    
    /* generation of code for scanning one tile */
    /*  generation of bounds */ 
    /* est completement inutile 
    for (pb = initial_basis; pb!=NULL; pb=pb->succ) {
	make_bound_expression(pb->var, initial_basis, sc_tile, &lower, &upper); 
    } */
  
    /* loop body generation */

    /* Compute the coordinate changes: there should be none for the time being
     * because we keep the initial basis to scan iterations within one tile, i.e
     * G must be the identity matrix
     */
    pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
    scanning_base_to_vect(G, n, tile_basis, pvg);
    pv1 = sc_tile_scan->inegalites->succ->vecteur;
    pv2 = vect_change_base(pv1, initial_basis, pvg);    

    l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
    lower = range_upper(loop_range(l));
    upper= expression_to_expression_newbase(lower, pvg, initial_basis);


   s_lhyp = code_generation(lls, pvg, initial_basis, tile_basis, sc_tile_scan);
/* essai avec new-basis au lieu de tile basis mais ne marche pas car itere 2 fois uniquement 
 tu as maintenant le parcours de l'interieur et des tiles mais plus celui des tiles*/
    s_lhyp = code_generation(lls, pvg, initial_basis, new_basis, sc_tile);
    
    debug(8," tiling","End\n");

    debug_off();

    return(s_lhyp);
}

bool
loop_tiling(string module_name)
{
    bool return_status = FALSE;

    return_status = interactive_loop_transformation(module_name, tiling);
    
    return return_status;
}
