/**
 * $Id: compress_parms.h,v 1.3 2006/03/07 04:23:26 loechner Exp $
 * @author B. Meister 12/2003-2006
 * LSIIT -ICPS 
 * UMR 7005 CNRS
 * Louis Pasteur University (ULP), Strasbourg, France
 */
#ifndef __BM_COMPRESS_PARMS_H__
#define __BM_COMPRESS_PARMS_H__

#include "matrix_addon.h"
#include "matrix_permutations.h"
#include <assert.h>


/* given a full-row-rank nxm matrix M(made of row-vectors), 
 computes the basis K (made of n-m column-vectors) of the integer kernel of M
 so we have: M.K = 0 */
Matrix * int_ker(Matrix * M);

/* Compute the overall period of the variables I for (MI) mod |d|,
 where M is a matrix and |d| a vector
 Produce a diagonal matrix S = (s_k) where s_k is the overall period of i_k */
Matrix * affine_periods(Matrix * M, Matrix * d);

/* given a matrix B' with m rows and m-vectors C' and d, computes the 
 basis of the integer solutions to (B'N+C') mod d = 0.
returns NULL if there is no integer solution */
Matrix * int_mod_basis(Matrix * Bp, Matrix * Cp, Matrix * d);

/* given a parameterized constraints matrix with m equalities, computes the
 compression matrix C such that there is an integer solution in the variables
 space for each value of N', with N = Cmp N' (N are the original parameters) */
Matrix * compress_parms(Matrix * E, int nb_parms);

/* extracts equalities involving only parameters */
Matrix * Constraints_Remove_parm_eqs(Matrix ** M, Matrix ** Ctxt, 
				     int renderSpace);
Polyhedron * Polyhedron_Remove_parm_eqs(Polyhedron ** P, Polyhedron ** C, 
					int renderSpace, int maxRays);

/* given a matrix of m parameterized equations, compress the parameters and
 transform the variable space into a n-m space. */
Matrix * full_dimensionize(Matrix const * M, int nb_parms, 
			   Matrix ** Validity_Lattice);

#endif /* __BM_COMPRESS_PARMS_H__ */
