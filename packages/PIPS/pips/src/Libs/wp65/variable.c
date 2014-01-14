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
 /* Code Generation for Distributed Memory Machines
  *
  * Define and allocate local variables as well as emulated shared variables
  *
  * File: variable.c
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin, Corinne Ancourt, Lei Zhou
  * 1991
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <limits.h>
#include <string.h>

#include "genC.h"
#include "misc.h"

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
/* for the phi variable */
#include "effects-generic.h"
#include "effects-convex.h"
#include "arithmetique.h"

#include "matrice.h"
#include "tiling.h"

#include "wp65.h"

/* loop_nest_to_local_variables():
 *
 * Key function allocating local copies according to interferences between
 * references. It also gather information about references (which is also
 * indirectly available in the effect information), allocate emulated
 * shared variables, and create different mappings between local variables
 * and references.
 */
void loop_nest_to_local_variables(
    initial_module, compute_module, memory_module, 
    llv_to_lcr, r_to_llv, v_to_lllv, r_to_ud,
    v_to_esv, v_to_nlv,
    lpv, body, indices, dg, bn, ls, pd, tile)
entity initial_module;  /* initial module: achtung, this arg. is not used!?! */
entity compute_module;  /* compute module */
entity memory_module;   /* memory module */
hash_table llv_to_lcr;	/* local variable to list of conflicting references */
hash_table r_to_llv;	/* reference to list of local variables  */
hash_table v_to_lllv;	/* variable to list of lists of local variables */
hash_table r_to_ud;	/* reference to use def */
hash_table v_to_esv;	/* variable to emulated shared variable  */
hash_table v_to_nlv;	/* variable to number of associated local
			   variables (local to this procedure) */
list lpv; /* private variables */
statement body;
Pbase indices;
graph dg;
int bn;
int ls;
int pd;
tiling tile;
{
    list block;
    instruction instr = statement_instruction(body);

    debug(6,"loop_nest_to_local_variables","begin\n");

    ifdebug(5) { 
	(void) fprintf(stderr,"Private variables:");
	print_arguments(lpv);
    } 

/*    pips_assert("loop_nest_to_local_variables", 
      instruction_block_p(instr));*/ 

    if (instruction_call_p(instr)) 
	block = CONS(STATEMENT,body,NIL);
    else block = instruction_block(instr);

    for( ;!ENDP(block); POP(block)) {
	statement s = STATEMENT(CAR(block));

	if(assignment_statement_p(s)) {
	    list lexpr =call_arguments(instruction_call(statement_instruction(s)));
	    /* first reference in statement s */
	    bool first_reference = true;
	    /* there are only two expressions to loop over: the lhs and the 
	       rhs */
	    for(; !ENDP(lexpr); POP(lexpr)) {
		expression e = EXPRESSION(CAR(lexpr));
		list lr = expression_to_reference_list(e, NIL);
		list consr;

		ifdebug(7) {
		    (void) fprintf(stderr, "reference list:");
		    print_reference_list(lr);
		    (void) fprintf(stderr, "first_reference=%s\n",
				   bool_to_string(first_reference));
		}

		for(consr = lr; !ENDP(consr) ; POP(consr)) {
		    reference r = REFERENCE(CAR(consr));
		    entity rv = reference_variable(r);

		    if(entity_is_argument_p(rv, lpv) || reference_indices(r) ==NIL) {
			debug(7,"loop_nest_to_local_variables",
			      "Variable %s is private\n", 
			      entity_local_name(rv));
			
			first_reference = false;
		    }
		    else {
			hash_put(r_to_ud,  r,
				 (void*)(intptr_t)(first_reference? 
					   is_action_write : is_action_read));
			first_reference = false;
			if(!reference_conflicting_test_and_update(r, dg, 
								  v_to_lllv, 
								  llv_to_lcr,
								  r_to_llv,
								  r_to_ud)) {
			    entity v = reference_variable(r);
			    intptr_t n;
			    list llv;
			    (void) find_or_create_emulated_shared_variable(v, memory_module, 
									   v_to_esv,
									   bn, ls);
			    if((n = (intptr_t) hash_get(v_to_nlv, (char *) v))
			       == (intptr_t) HASH_UNDEFINED_VALUE)
				n = 0;
			    llv = make_new_local_variables(v, compute_module, n, pd, 
							   v_to_lllv);
			    hash_put(llv_to_lcr,  llv, 
				      CONS(REFERENCE, r, NIL));
			    hash_put(r_to_llv,  r,  llv);
			    hash_put(v_to_nlv,  v,  (void*)(n+1));
			}
		    }
		}
		gen_free_list(lr);
	    }
	}
    }
    /* FI: local variable dimensions should be set later by the caller
       so as to use the clustering performed here to compute space
       complexity, to use space complexity to define the tiling and
       to use the tiling to define the dimensions */
    set_dimensions_of_local_variables(v_to_lllv, indices, tile, llv_to_lcr);

    debug(6,"loop_nest_to_local_variables","end\n");
}

entity make_emulated_shared_variable(v, memory_module, bn, ls)
entity v;
entity memory_module;
int bn;
int ls;
{
    string esv_name = strdup(concatenate(entity_local_name(memory_module),
					 MODULE_SEP_STRING,
					 EMULATED_SHARED_MEMORY_PREFIX,
					 entity_local_name(v),
					 NULL));
    entity esv = gen_find_tabulated(esv_name, entity_domain);
    type tv = entity_type(v);
    basic bv = variable_basic(type_variable(tv));
    list ldv = variable_dimensions(type_variable(tv));
    entity a;
    int number_of_elements = 1;

    dimension esvd1 = dimension_undefined;
    dimension esvd2 = dimension_undefined;
    debug(8,"make_emulated_shared_variable", "begin\n");

    pips_assert("make_emulated_shared_variable", esv == entity_undefined );
    pips_assert("make_emulated_shared_variable", type_variable_p(tv));

    

    esv = make_entity(esv_name,
		      type_undefined,	/* filled in the following */
		      storage_undefined,	
		      value_undefined ); /* filled in the following */

    /* generate the proper type; basic is preserved but the array is made
       two dimensional */

    for( ; !ENDP(ldv); POP(ldv)) {
	dimension d = DIMENSION(CAR(ldv));
	int size = dimension_size(d);
	number_of_elements *= size;
    }

    ifdebug(8) {
	(void) fprintf(stderr,"number of elements for %s: %d\n", 
		       entity_name(esv), number_of_elements);
    }

     /* In two sprintf , -1 is added once seperately  by LZ
      * make_expression_1 takes the place of the make_expression_1
      * 12/11/91
      */
    if(number_of_elements > 1) {
	esvd1 = make_dimension(int_to_expression(0),
			       int_to_expression(ls-1));
	esvd2 = make_dimension(int_to_expression(0),
			       int_to_expression((number_of_elements+ls*bn-1)/(ls*bn)));

	entity_type(esv) = MakeTypeVariable(copy_basic(bv), 
					    CONS(DIMENSION, esvd1,
						 CONS(DIMENSION, esvd2, NIL)));
    }
    else {
	/* FI->CA, LZ: what should we do with scalar variables? Put a copy
	   on each memory bank? Take into account memory bank conflicts
	   shown in thresholding (PUMA WP 6.1 and 6.7) */
	entity_type(esv) = MakeTypeVariable( copy_basic(bv), NIL);
    }

    entity_initial(esv) = make_value_unknown();

    a = FindEntity(module_local_name(memory_module), 
			      DYNAMIC_AREA_LOCAL_NAME);
    entity_storage(esv)=make_storage(is_storage_ram,
				     (make_ram(memory_module, 
					       a,
					       add_variable_to_area(a,
								    esv),
					       NIL)));
    
    AddEntityToDeclarations(esv,memory_module);

    debug(8,"make_emulated_shared_variable", "esv_name=%s\n", entity_name(esv));
    ifdebug(8) print_sentence(stderr,sentence_variable(esv, NIL));
    debug(8,"make_emulated_shared_variable", "end\n");

    return(esv);
}

entity find_or_create_emulated_shared_variable(v, memory_module, v_to_esv, bn,ls)
entity v;
entity memory_module;
hash_table v_to_esv;
int bn;
int ls;
{
    entity esv = entity_undefined;

    if ( (esv = gen_find_tabulated(concatenate(entity_local_name(memory_module),
					     MODULE_SEP_STRING,
					     EMULATED_SHARED_MEMORY_PREFIX,
					     entity_local_name(v),
					     NULL),
				 entity_domain)) != entity_undefined )
	/* nothing to do */
	;
    else {
	esv = make_emulated_shared_variable(v, memory_module, bn, ls);
	hash_put(v_to_esv, (char *) v, (char *) esv);
    }
    return (esv);
}



list make_new_local_variables(v, compute_module, number, pd, v_to_lllv)
entity v;
entity compute_module;
int number;     /* local copy number */
int pd;         /* pipeline depth */
hash_table v_to_lllv;
{
    /* Local variables cannot be immediately dimensionned because this
       will depend on *all* associated references.
       
       To avoid a possible "fusion" between references connected in the dg,
       local variable declarations should be delayed to that point. */

    /* concatenate cannot be used (directly) because of n and s conversions */
    char *local_variable_name;
    int s;
    list llv = NIL;
    list lllv = NIL;
    const char* computational_name = entity_local_name(compute_module);

    type tv = entity_type(v);
    entity a; 

    debug(7,"make_new_local_variables","begin v=%s, number=%d, pd=%d\n",
	  entity_name(v), number, pd);

    for(s=0; s<pd; s++) {
	entity lv;
	/* a new btv is necessary for each variable because CONS cannot
	   be shared under NewGen rules */
	basic btv =  copy_basic(variable_basic(type_variable(tv)));

	(void) asprintf(&local_variable_name,"%s%s%s%s%s%d%s%d",
		       computational_name,
		       MODULE_SEP_STRING,
		       LOCAL_MEMORY_PREFIX,
		       entity_local_name(v),
		       LOCAL_MEMORY_SEPARATOR,
		       number,
		       LOCAL_MEMORY_SEPARATOR,
		       s);
		
	/* FI->LZ: the type should be v's type, except for the dimensions,
	   the storage should be RAM and allocated in the dynamic
	   area of module_name, and the value is undefined;
	   the actual dimensions will be updated later */
	lv = make_entity(local_variable_name,
			 MakeTypeVariable(btv, NIL),
			 storage_undefined,
			 value_undefined);

	a = FindEntity(module_local_name(compute_module), 
				  DYNAMIC_AREA_LOCAL_NAME);
	pips_assert("make_new_local_variables",!entity_undefined_p(a));

	entity_storage(lv) = make_storage(is_storage_ram,
					  (make_ram(compute_module, a,
						    add_variable_to_area(a, lv),
						    NIL)));


	AddEntityToDeclarations( lv,compute_module);
	llv = gen_nconc(llv, CONS(ENTITY, lv, NIL));
    }

    if((lllv = (list) hash_get(v_to_lllv, (char *) v)) 
       == (list) HASH_UNDEFINED_VALUE) {
	lllv = CONS(LIST, llv, NIL);
	hash_put(v_to_lllv, (char *) v, (char *) lllv);
    }
    else {
	/* update v_to_lllv by side effect */
	lllv = gen_nconc(lllv, CONS(LIST, llv, NIL));
    }


    debug(7,"make_new_local_variables","end\n");
    
    return llv;
}

/* reference_conflicting_test_and_update():
 *
 * Build, incrementally, connected component in the dependence graph.
 * Input dependences *should* be used to save local memory space.
 * Mappings are updated.
 *
 * Too many local variables might be created if two connected components
 * had to be fused. This is not likely to happen with real code. In that
 * case, the function would stop with pips_error().
 */
bool reference_conflicting_test_and_update(r, dg, 
					      v_to_lllv, llv_to_lcr,
					      r_to_llv, r_to_ud)
reference r;
graph dg;
hash_table v_to_lllv;
hash_table llv_to_lcr;
hash_table r_to_llv;
hash_table r_to_ud;
{
    list current_llv = list_undefined;
    entity rv;				/* referenced variable */
    list lllv;
    list cllv;
    bool conflicting = false;

    debug(8,"reference_conflicting_test_and_update", "begin\n");

    rv = reference_variable(r);

    ifdebug(8) {
	(void) fprintf(stderr, "reference %p to %s:", 
		       r, entity_local_name(rv));
	print_words(stderr, words_reference(r, NIL));
	(void) putc('\n', stderr);
    }

    if((lllv = (list) hash_get(v_to_lllv, (char *) rv)) 
       == (list) HASH_UNDEFINED_VALUE) {
	debug(8, "reference_conflicting_test_and_update", 
	      "no local variables for reference %p to %s\n",
	      r, entity_local_name(rv));

	debug(8,"reference_conflicting_test_and_update", "return FALSE\n");
	return false;
    }

    for(cllv = lllv; !ENDP(cllv); POP(cllv)) {
	list llv = LIST(CAR(cllv));
	list lcr = (list) hash_get(llv_to_lcr, (char *) llv);
	list ccr;

	for(ccr = lcr; !ENDP(ccr); POP(ccr)) {
	    reference r2 = REFERENCE(CAR(ccr));
	    entity rv2 = reference_variable(r2);

	    ifdebug(8) {
		(void) fprintf(stderr, "conflict_p with reference %p to %s:",
			       r2, entity_local_name(rv2));
		print_words(stderr, words_reference(r2, NIL));
		(void) putc('\n', stderr);
	    }

	    if(reference_equal_p(r, r2)) {
		if(hash_get(r_to_ud,(char*) r)==hash_get(r_to_ud,(char*) r2)){
		/* short cut for identical references; not necessarily a good
		   idea for the body translation process; design to save
		   time in transfer code generation */
		/* should go one step forwards and take care of use-def,
		   easier by introducing a "use and def" value or by adding
		   the new reference directly */

		    debug(8,"reference_conflicting_test_and_update", 
			  "reference equality and same use, return TRUE\n");
		}
		else {
		    /* we need to remember there is a use and a def */
		    /* this is very clumsy as we may keep a large number
		       of identical def after one use and vice-versa;
		       we need a use-and-def value or we have to keep
		       all references and filter them for data movements */
		    /* update llv_to_lcr by side-effect */
		    lcr = gen_nconc(lcr, CONS(REFERENCE, r, NIL));
		    debug(8,"reference_conflicting_test_and_update", 
			  "reference equality and same use, return TRUE\n");
		}
		hash_put(r_to_llv, (char *) r, (char *) llv);
		return true;
	    }
	    else if(reference_conflicting_p(r, r2, dg)) {
		if(conflicting) {
		    /* should not happen too often: the regions associated 
		       with two references do not intersect together but
		       each of them intersect a third one on which we bump
		       afterwards; the two initial llv's should be merged...
		       */
		    pips_assert("reference_conflicting_test_and_update",
				current_llv != llv);
		    pips_internal_error("local variable merge not implemented");
		}
		else {
		    conflicting = true;
		    /* update llv_to_lcr by side-effect */
		    lcr = gen_nconc(lcr, CONS(REFERENCE, r, NIL));
		    hash_put(r_to_llv, (char *) r, (char *) llv);
		    /* save current_llv for a future (?) merge */
		    current_llv = llv;
		    /* no need to study conflicts with other references
		       in the *same* connected component */
		debug(8,"reference_conflicting_test_and_update", 
		      "Conflict! Look for conflicts with other components\n");
		    break;
		}
	    }
	}
    }

    debug(8,"reference_conflicting_test_and_update", "return %d\n",
	  conflicting);
    return conflicting;
}

bool reference_conflicting_p(r1,r2,dg)
reference r1,r2;
graph dg;
{
    /*
      graph = vertices:vertex* ;
      successor = arc_label x vertex ;
      vertex = vertex_label x successors:successor* ;
      */
    list ver1 = graph_vertices(dg);

    debug(8,"reference_conflicting_p","begin\n");
    ifdebug(8) {
	(void) fprintf(stderr,"Reference 1 %p: ", r1);
	print_words(stderr, words_reference(r1, NIL));
	(void) fprintf(stderr,"\nReference 2 %p: ", r2);
	print_words(stderr, words_reference(r2, NIL));
	(void) fputc('\n',stderr);
    }

    MAPL(pm1,{
	vertex v1 = VERTEX(CAR(pm1));
	list ver2 = vertex_successors(v1);
	MAPL(pm2,{
	    successor su = SUCCESSOR(CAR(pm2));

	    dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
	    list conf_list = dg_arc_label_conflicts(dal);

	    MAPL(pm3,{
		conflict conf2 = CONFLICT(CAR(pm3));

		effect e_source = conflict_source(conf2);
		effect e_sink = conflict_sink(conf2);

		reference r11 = effect_any_reference(e_source);
		reference r21 = effect_any_reference(e_sink);

		ifdebug(8) {
		    (void) fprintf(stderr,"Test with reference 1 %p: ", r1);
		    print_words(stderr, words_reference(r1, NIL));
		    (void) fprintf(stderr," and reference 2 %p: ", r2);
		    print_words(stderr, words_reference(r2, NIL));
		    (void) fputc('\n',stderr);
		    (void) fprintf(stderr,"Test with reference 11 %p: ", r11);
		    print_words(stderr, words_reference(r11, NIL));
		    (void) fprintf(stderr," and reference 21 %p: ", r21);
		    print_words(stderr, words_reference(r21, NIL));
		    (void) fputc('\n',stderr);
		    (void) fputc('\n',stderr);
		}

		if( (reference_equal_p(r11,r1) &&  reference_equal_p(r21,r2))
		   || (reference_equal_p(r11,r2) && reference_equal_p(r21,r1)) ) {
		    debug(8,"reference_conflicting_p","return TRUE\n");
		    return true;
		}
	    },conf_list);
	},ver2);
    },ver1);

    debug(8,"reference_conflicting_p","return FALSE\n");
    return false;
}

/* Psysteme make_tile_constraints(P, b):
 *
 * convert a partitioning matrice into a system of linear constraints
 * for a tile whose origin is 0, i.e. generate constraints over local
 * indices:
 *
 * ->      -1 ->   ---->
 * 0 <= k P   b <= (k-1)
 *
 * where k is the denominator of P inverse.
 *
 * FI: such functions should be put together in a file, linear.c
 */
Psysteme make_tile_constraints(P, b)
matrice P;
Pbase b;
{
    Psysteme tc = sc_new();

    int d = base_dimension(b);
    Value k = DENOMINATOR(P);
    int i,j;

    matrice IP = matrice_new(d,d);
    Pcontrainte c1;

    debug(8,"make_tile_constraints", "begin\n");

    pips_assert("make_tile_constraints", value_one_p(k));

    ifdebug(8) {
	(void) fprintf(stderr,"Partitioning matrix P:\n");
	matrice_fprint(stderr, P, d, d);
    }

    matrice_general_inversion(P, IP, d);
    matrice_normalize(IP, d, d);
    k = DENOMINATOR(IP);

  /*  pips_assert("make_tile_constraints", k > 1); */

    ifdebug(8) {
	(void) fprintf(stderr,"Inverse of partitioning matrix, IP:\n");
	matrice_fprint(stderr, IP, d, d);
    }

    for ( i=1; i<=d; i++) {
	Pcontrainte c = contrainte_new();

	for ( j=1; j<=d; j++) {
	    vect_add_elem(&contrainte_vecteur(c), 
			  variable_of_rank(b,i),
			  value_uminus(ACCESS(IP, d, j, i)));

	}
        sc_add_inegalite(tc, c);
	c1 = contrainte_dup(c);
	contrainte_chg_sgn(c1);
	vect_add_elem(&contrainte_vecteur(c1), TCST, value_minus(VALUE_ONE,k));
	sc_add_inegalite(tc, c1);
    }

    sc_creer_base(tc);
    matrice_free(IP);

    ifdebug(8) {
	sc_fprint(stderr, tc, (string(*)(Variable))entity_local_name);
    }

    debug(8,"make_tile_constraints", "end\n");

    return (tc);
}

void set_dimensions_of_local_variables(v_to_lllv, basis, tile, llv_to_lcr)
hash_table v_to_lllv;
Pbase basis;
tiling tile;
hash_table llv_to_lcr;
{
    Psysteme tc = SC_UNDEFINED;
    matrice P = (matrice) tiling_tile(tile);

    debug(8,"set_dimensions_of_local_variables","begin\n");

    tc = make_tile_constraints(P, basis);

    HASH_MAP(v, clllv, {
	list cllv;
	for(cllv = (list) clllv; !ENDP(cllv); POP(cllv)) {
	    list llv = LIST(CAR(cllv));
	    list lr = (list) hash_get(llv_to_lcr, (char *) llv);
	    set_dimensions_of_local_variable_family(llv, tc, lr,tile,vect_size(basis));
	}
    }, v_to_lllv);

    sc_rm(tc);

    debug(8,"set_dimensions_of_local_variables","end\n");
}

/* void set_dimensions_of_local_variable_family(llv, tc, lr):
 *
 * The algorithm used is not general; references are assumed equal up to
 * a translation;
 *
 * A general algorithm, as would be necessary for instance if a dependence
 * was detected in the transposition algorithm between M(I,J) and M(J,I),
 * would preserve some translation information for each reference in lr
 * so as to generate proper new references to the local variable.
 */
void set_dimensions_of_local_variable_family(llv, tc, lr,tile,dimn)
list llv; /* list of local variables with same dimensions used at
	    different pipeline stages */
Psysteme tc;
list lr; /* non-empty list of associated references */
tiling tile;
int dimn;
{
    /* let's use the first reference to find out the number of dimensions */
    reference r;
    entity rv;			/* referenced variable */
    type rvt;			/* referenced variable type */
    list rvld;			/* referenced variable dimension list */
    int d = -1;			/* dimension number */
    list lvd = NIL;		/* dimensions for the local variables */
    bool first_ref;
    matrice P = (matrice) tiling_tile(tile);
    debug(8,"set_dimensions_of_local_variable_family","begin\n");

    r = REFERENCE(CAR(lr));
    rv = reference_variable(r);
    rvt = entity_type(rv);
    rvld = variable_dimensions(type_variable(rvt));

    debug(8,"set_dimensions_of_local_variable_family","entity=%s\n",
	  entity_name(rv));

    for( d = 1; !ENDP(rvld); POP(rvld), d++) {
	Value imax = VALUE_MIN;
	Value gmin = VALUE_MAX;
	Value gmax = VALUE_MIN;
	list cr = list_undefined;
	dimension dimd = dimension_undefined;
	

	/* computation of the initial bounds of Entity rv */
	dimension dim1 = DIMENSION(CAR(rvld));
	expression lower= dimension_lower(dim1);
	normalized norm1 = NORMALIZE_EXPRESSION(lower);
	expression upper= dimension_upper(dim1);
	normalized norm2 = NORMALIZE_EXPRESSION(upper);
	if (normalized_linear_p(norm1) && normalized_linear_p(norm2)) {
	    gmin = vect_coeff(TCST,(Pvecteur) normalized_linear(norm1));
	    value_decrement(gmin);
	    gmax = vect_coeff(TCST,(Pvecteur) normalized_linear(norm2));
	    value_decrement(gmax);
	    imax = gmax;
	}
	first_ref = true;
	for(cr = lr; !ENDP(cr); POP(cr)) {
	    entity phi = make_phi_entity(d);
	    expression e;
	    normalized n;
	    Pvecteur vec;
	    Pcontrainte eg;
	    Psysteme s;
	    Value min, max, coef;
	    r = REFERENCE(CAR(cr));
	    e = find_ith_expression(reference_indices(r), d);
	    n = NORMALIZE_EXPRESSION(e);

	    /* pips_assert("set_dimensions_of_local_variable_family", */
	    if (normalized_linear_p(n)) {
		vec = vect_dup((Pvecteur) normalized_linear(n));
		vect_add_elem(&vec, (Variable) phi, VALUE_MONE);
		eg = contrainte_make(vec);

		/* pour tenir compte des offsets numeriques dans les 
		   fonctions d'acces */
		coef =vect_coeff(TCST,vec);
		if (value_pos_p(coef)) value_addto(gmax,coef);
		else value_addto(gmin,coef);

		s = sc_dup(tc);
		sc_add_egalite(s, eg);
		vect_add_variable(sc_base(s), (Variable) phi);

		s = sc_normalize(s);
		ifdebug(8) {
		    (void) fprintf(stderr,
				   "System on phi for dimension %d:\n", d);
		    sc_fprint(stderr, s, (string(*)(Variable))entity_local_name);
		}


		if(!sc_minmax_of_variable(s, (Variable) phi, 
					  &min, &max))
		    pips_internal_error("empty domain for phi");

		if(value_min_p(min) || value_max_p(max)) {
		    Value divis= ACCESS(P,dimn,d,d);
		    /* parameter ==> min = max = 1 */
		    /*pips_internal_error("unbounded domain for phi, %s",
		      "check tile bounds and subscript expressions"); */
		    min= VALUE_ZERO;
		    max= value_lt(divis,VALUE_CONST(999)) && 
			value_gt(divis,VALUE_ONE)? value_div(imax,divis): imax;
		}

		debug(8,"set_dimensions_of_local_variable_family",
		      "bounds for dimension %d: [%d:%d]\n", d, min, max);

		gmin = first_ref? value_max(gmin, min): value_min(gmin, min);
		gmax = first_ref? value_min(gmax, max): value_max(gmax, max);
		
		if (value_gt(gmin,gmax)) gmax = gmin;
		first_ref=false;		
	    }
	}
	dimd = make_dimension(int_to_expression(VALUE_TO_INT(gmin)),
			      int_to_expression(VALUE_TO_INT(gmax)));

	debug(8,"set_dimensions_of_local_variable_family",
	      "bounds for dimension %d: [%d:%d]\n", d, gmin, gmax);

	lvd = gen_nconc(lvd, CONS(DIMENSION, dimd, NIL));
    }

    /* update types */
    MAPL(clv, 
     {
	 entity lv = ENTITY(CAR(clv));
	 type tlv = entity_type(lv);
	 variable tv = type_variable(tlv);

	 /* sharing is not legal under NewGen rules; to avoid
	    it lvd is duplicated at the CONS level, except the first time */
	 if(clv!=llv)
	     lvd = gen_copy_seq(lvd);
	 variable_dimensions(tv) = lvd;
     },
	 llv);

    ifdebug(8) {
	MAPL(clv, 
	 {
	     entity lv = ENTITY(CAR(clv));
	     print_sentence(stderr,sentence_variable(lv, NIL));
	 },
	     llv);
    }

    pips_debug(8,"end\n");
}
