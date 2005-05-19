// Polylib Matrix addons
// Mainly deals with polyhedra represented as a matrix (implicit form)
// B. Meister

#ifndef __BM_MATRIX_ADDON_H__
#define __BM_MATRIX_ADDON_H__

#include<polylib/polylib.h>
#include<assert.h>


#define show_matrix(M) {printf(#M"= \n"); Matrix_Print(stderr,P_VALUE_FMT,(M));}

// B_Lcm is Maybe not at the right place here, but...
/* computes c = lcm(a,b) using Gcd(a,b,&c) */
void B_Lcm(Value a, Value b, Value * c);

// splits a matrix of constraints M into a matrix of equalities Eqs and a matrix of inequalities Ineqs
// allocs the new matrices.
void split_constraints(Matrix const * M, Matrix ** Eqs, Matrix **Ineqs);

// returns the dim-dimensional identity matrix
Matrix * Identity_Matrix(unsigned int dim);

// given a n x n integer transformation matrix transf, compute its inverse M/g, where M is a nxn integer matrix.
// g is a common denominator for elements of (transf^{-1})
unsigned int mtransformation_inverse(Matrix * transf, Matrix ** inv);

// simplify a matrix seen as a polyhedron, by dividing its rows by the gcd of their elements.
void mpolyhedron_simplify(Matrix * polyh);

// inflates a polyhedron (represented as a matrix) P, so that the apx of its Ehrhart Polynomial is an upper bound of the Ehrhart polynomial of P
// WARNING: this inflation is supposed to be applied on full-dimensional polyhedra.
void mpolyhedron_inflate(Matrix * polyh, unsigned int nb_parms);

// deflates a polyhedron (represented as a matrix) P, so that the apx of its Ehrhart Polynomial is a lower bound of the Ehrhart polynomial of P
// WARNING: this deflation is supposed to be applied on full-dimensional polyhedra.
void mpolyhedron_deflate(Matrix * polyh, unsigned int nb_parms);

// use an eliminator row to eliminate a variable in a victim row (without changing the sign of the victim row -> important if it is an inequality).
void eliminate_var_with_constr(Matrix * Eliminator, unsigned int eliminator_row, Matrix * Victim, unsigned int victim_row, unsigned int var_to_elim);



// PARTIAL MAPPINGS

// compress the last vars/pars of the polyhedron M expressed as a polylib matrix
// - adresses the full-rank compressions only
// - modfies M
void mpolyhedron_compress_last_vars(Matrix * M, Matrix * compression);

// use a set of m equalities Eqs to eliminate m variables in the polyhedron Ineqs represented as a matrix
// eliminates the m first variables
// - assumes that Eqs allows to eliminate the m equalities
// - modifies Ineqs
unsigned int mpolyhedron_eliminate_first_variables(Matrix * Eqs, Matrix * Ineqs);


#endif // __BM_MATRIX_ADDON_H__
