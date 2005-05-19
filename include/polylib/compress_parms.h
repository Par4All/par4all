// (C) B. Meister 12/2003
// LSIIT -ICPS 
// UMR 7005 CNRS
// Louis Pasteur University (ULP), Strasbourg, France

#ifndef __BM_COMPRESS_PARMS_H__
#define __BM_COMPRESS_PARMS_H__

#include "matrix_addon.h"
#include "matrix_permutations.h"
#include<assert.h>
// #include "apx.h"

//computes the modulo inverse of a modulo n
Value modulo_inverse(const Value a, const Value n);

// given a full-row-rank nxm matrix M(made of row-vectors), 
// computes the basis K (made of n-m column-vectors) of the integer kernel of M
// so we have: M.K = 0
Matrix * int_ker(Matrix * M);

// Compute the overall period of the variables I for (MI) mod |d|,
// where M is a matrix and |d| a vector
// Produce a diagonal matrix S = (s_k) where s_k is the overall period of i_k
Matrix * affine_periods(Matrix * M, Matrix * d);

// given a matrix B' with m rows and m-vectors C' and d, computes the 
// basis of the integer solutions to (B'N+C') mod d = 0.
// returns NULL if there is no integer solution
Matrix * int_mod_basis(Matrix * Bp, Matrix * Cp, Matrix * d);

// given a parameterized constraints matrix with m equalities, computes the compression matrix 
// C such that there is an integer solution in the variables space for each value of N', with 
// N = Cmp N' (N are the original parameters)
Matrix * compress_parms(Matrix * E, int nb_parms);

// given a matrix of m parameterized equations, compress the parameters and transform the variable space into a n-m space.
Matrix * full_dimensionize(Matrix const * M, int nb_parms, Matrix ** Validity_Lattice);

// computes the compression matrices G and G2 so that: 
// - with G, I is integer for any integer N
// - with G2, I2 is integer for any integer N, I1
// output: the resulting compression
Matrix * compress_for_full_dim(Matrix * M, int nb_parms, Matrix **  compression, Matrix ** G);
Matrix * compress_to_full_dim2(Matrix * Eqs, Matrix * Ineqs, int nb_parms, unsigned int * vars_to_keep, Matrix ** compression, Matrix ** Cmp_Eqs);

Matrix * eliminate_last_variables(Matrix * Eqs, Matrix * Ineqs, int nb_vars);

// takes a transformation matrix, and expands it to a higher dimension with the identity matrix 
Matrix * expand_left_to_dim(Matrix * M, int new_dim);

#endif // __BM_COMPRESS_PARMS_H__
