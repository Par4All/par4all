/* B. Meister 12/2003-2005
 LSIIT -ICPS 
 UMR 7005 CNRS
 Louis Pasteur University (ULP), Strasbourg, France 
*/

#include <polylib/polylib.h>

/* given a full-row-rank nxm matrix M(made of m row-vectors), 
 computes the basis K (made of n-m column-vectors) 
 of the integer kernel of the rows of M
 so we have: M.K = 0 */
Matrix * int_ker(Matrix * M) {
  Matrix *U, *Q, *H, *H2, *K;
  int i, j, rk;

  /* eliminate redundant rows : UM = H*/
  right_hermite(M, &H, &Q, &U);
  for (rk=H->NbRows-1; (rk>=0) && Vector_IsZero(H->p[rk], H->NbColumns); rk--);
  rk++;
    
  /* there is a non-null kernel if and only if the dimension m of 
     the space spanned by the rows 
     is inferior to the number n of variables */
  if (M->NbColumns <= rk) {
    K = Matrix_Alloc(M->NbColumns, 0);
    return K;
  }
  Matrix_Free(U); 
  Matrix_Free(Q);
  /* fool left_hermite  by giving NbRows =rank of M*/
  H->NbRows=rk;
  /* computes MU = [H 0] */
  left_hermite(H, &H2, &Q, &U); 
  H->NbRows==M->NbRows;
  Matrix_Free(H);
  /* obviously, the Integer Kernel is made of the last n-rk columns of U */
  K=Matrix_Alloc(U->NbRows, U->NbColumns-rk);
  for (i=0; i< U->NbRows; i++)
    for(j=0; j< U->NbColumns-rk; j++) 
      value_assign(K->p[i][j], U->p[i][rk+j]);

  /* clean up */
  Matrix_Free(H2);
  Matrix_Free(U);
  Matrix_Free(Q);
  return K;
}

/* Compute the overall period of the variables I for (MI) mod |d|,
 where M is a matrix and |d| a vector
 Produce a diagonal matrix S = (s_k) where s_k is the overall period of i_k */
Matrix * affine_periods(Matrix * M, Matrix * d) {
  Matrix * S;
  unsigned int i,j;
  Value tmp;
  Value * periods = (Value *)malloc(sizeof(Value) * M->NbColumns);
  value_init(tmp);
  for(i=0; i< M->NbColumns; i++) {
    value_init(periods[i]);
    value_set_si(periods[i], 1);
  }
  for (i=0; i<M->NbRows; i++) {
    for (j=0; j< M->NbColumns; j++) {
      Gcd(d->p[i][0], M->p[i][j], &tmp);
      value_division(tmp, d->p[i][0], tmp);
      Lcm3(periods[j], tmp, &(periods[j]));
     }
  }
  value_clear(tmp);

  /* 2- build S */
  S = Matrix_Alloc(M->NbColumns, M->NbColumns);
  for (i=0; i< M->NbColumns; i++) 
    for (j=0; j< M->NbColumns; j++)
      if (i==j) value_assign(S->p[i][j],periods[j]);
      else value_set_si(S->p[i][j], 0);

  /* 3- clean up */
  for(i=0; i< M->NbColumns; i++) value_clear(periods[i]);
  free(periods);
  return S;
}

/* 
given a matrix B' with m rows and m-vectors C' and d, computes the 
 basis of the integer solutions to (B'N+C') mod d = 0 (1).
 the Ns verifying the system B'N+C' = 0 are solutions of (1)
 K is a basis of the integer kernel of B: its column-vectors link two solutions of (1)
 Moreover, B'_iN mod d is periodic of period (s_ik):
 B'N mod d is periodic of period (s_k) = lcm_i(s_ik)
 The linear part of G is given by the HNF of (K | S), 
 where S is the full-dimensional diagonal matrix (s_k)
 the constant part of G is a particular solution of (1)
 if no integer constant part is found, there is no solution and this function returns NULL. 
*/
Matrix * int_mod_basis(Matrix * Bp, Matrix * Cp, Matrix * d) {
  int nb_eqs = Bp->NbRows;
  unsigned int nb_parms=Bp->NbColumns;
  unsigned int i, j;
  Matrix *H, *U, *Q, *M, *inv_H_M, *Ha, *Np_0, *N_0, *G, *K, *S, *KS;

  /*   a/ compute K and S */
  /* simplify the constraints */
  for (i=0; i< Bp->NbRows; i++)
    for (j=0; j< Bp->NbColumns; j++) 
      value_pmodulus(Bp->p[i][j], Bp->p[i][j], d->p[0][i]);
  K = int_ker(Bp);
  S = affine_periods(Bp, d);
  // show_matrix(K);
  // show_matrix(S);
  
  /*   b/ compute the linear part of G : HNF(K|S) */

  /* fill K|S */
  KS = Matrix_Alloc(nb_parms, K->NbColumns+ nb_parms);
  for(i=0; i< KS->NbRows; i++) {
    for(j=0; j< K->NbColumns; j++) value_assign(KS->p[i][j], K->p[i][j]);
    for(j=0; j< S->NbColumns; j++) value_assign(KS->p[i][j+K->NbColumns], S->p[i][j]);
  }

  // show_matrix(KS);

  /* HNF(K|S) */
  left_hermite(KS, &H, &U, &Q);
  Matrix_Free(KS);
  Matrix_Free(U);
  Matrix_Free(Q);
  
  // printf("HNF(K|S) = ");show_matrix(H);

  /* put HNF(K|S) in the p x p matrix S 
     (which has already the appropriate size so we spare a Matrix_Alloc) */
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++) 
      value_assign(S->p[i][j], H->p[i][j]);


  }
  Matrix_Free(H);

  /*   c/ compute U_M.N'_0 = N_0: */
  M = Matrix_Alloc(nb_eqs, nb_parms+nb_eqs);
  /* N'_0 = M_H^{-1}.(-C'), which must be integer
     and where H_M = HNF(M) with M = (B' D) : M.U_M = [H_M 0] */

  /*      copy the B' part */
  for (i=0; i< nb_eqs; i++) {
    for (j=0; j< nb_parms; j++) {
      value_assign(M->p[i][j], Bp->p[i][j]);
    }
    /*    copy the D part */
    for (j=0; j< nb_eqs; j++) {
      if (i==j) value_assign(M->p[i][j+nb_parms], d->p[i][0]);
      else value_set_si(M->p[i][j+nb_parms], 0);
    }
  }
  
  //       compute inv_H_M, the inverse of the HNF H of M = (B' D)
  left_hermite(M, &H, &Q, &U);
  inv_H_M=Matrix_Alloc(nb_eqs, nb_eqs+1);
  /* again, do a square Matrix from H, using the non-used Matrix Ha */
  Ha = Matrix_Alloc(nb_eqs, nb_eqs);
  for(i=0; i< nb_eqs; i++) {
    for(j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j], H->p[i][j]); 
    }
  }
  MatInverse(Ha, inv_H_M);
  Matrix_Free(Ha);
  Matrix_Free(H);
  Matrix_Free(Q); /* only inv_H_M and U_M (alias U) are needed */

  /*       compute (-C') */
  for (i=0; i< nb_eqs; i++) {
    value_oppose(Cp->p[i][0], Cp->p[i][0]);
  }

  /* Compute N'_0 = inv_H_M.(-C')
     actually compute (N' \\ 0) such that N = U^{-1}.(N' \\ 0) */
  Np_0 = Matrix_Alloc(U->NbColumns, 1);
  for(i=0; i< nb_eqs; i++) 
    {
      value_set_si(Np_0->p[i][0], 0);
      for(j=0; j< nb_eqs; j++) {
	value_addmul(Np_0->p[i][0], inv_H_M->p[i][j], Cp->p[j][0]);
      }
    }
  for(i=nb_eqs; i< U->NbColumns; i++) value_set_si(Np_0->p[i][0], 0);
  

  /* it is still needed to divide the rows of N'_0 by the common 
     denominator of the rows of H_M. If these rows are not divisible, 
     there is no integer N'_0 so return NULL */
  for (i=0; i< nb_eqs; i++) {

#ifdef GNUMP
    if (mpz_divisible_p(Np_0->p[i][0], inv_H_M->p[i][nb_eqs])) 
      mpz_divexact(Np_0->p[i][0], Np_0->p[i][0], inv_H_M->p[i][nb_eqs]);
#else
    if (!(Np_0->p[i][0]%inv_H_M->p[i][nb_eqs])) 
      Np_0->p[i][0]/=inv_H_M->p[i][nb_eqs];
#endif
    else {
      Matrix_Free(S);
      Matrix_Free(inv_H_M);
      Matrix_Free(Np_0);
      fprintf(stderr, "int_mod_basis > "
              "No particular solution: polyhedron without integer points.\n");
      return NULL;
    }
  }
  // show_matrix(Np_0);

  /* now compute the actual particular solution N_0 = U_M. N'_0 */
  N_0 = Matrix_Alloc(U->NbColumns, 1);
  /* OPT: seules les nb_eq premières valeurs de N_0 sont utiles en fait. */
  Matrix_Product(U, Np_0, N_0);
  // show_matrix(N_0);
  Matrix_Free(Np_0);
  Matrix_Free(U);

  /* build the whole compression matrix:  */
  G = Matrix_Alloc(S->NbRows+1, S->NbRows+1);
  for (i=0; i< S->NbRows; i++) {
    for(j=0; j< S->NbRows; j++) 
      value_assign(G->p[i][j], S->p[i][j]);
    value_assign(G->p[i][S->NbRows], N_0->p[i][0]);
  }

  for (j=0; j< S->NbRows; j++) value_set_si(G->p[S->NbRows][j],0);
  value_set_si(G->p[S->NbRows][S->NbRows],1);

  /* clean up */
  Matrix_Free(S);
  Matrix_Free(N_0);
  return G;
} /* int_mod_basis */


/* 
utility function: given a matrix containing the equations AI+BN+C=0, 
 compute the HNF of A : A = [Ha 0].Q and return :  
 . B'= H^-1.(-B) 
 . C'= H^-1.(-C)
 . U = Q^-1 (-> return value)
 . D, where Ha^-1 = D^-1.H^-1 with H and D integer matrices 
 in fact, as D is diagonal, we return d, a column-vector 
*/
Matrix * extract_funny_stuff(Matrix const * E, int nb_parms, 
			     Matrix ** Bp, Matrix **Cp, Matrix **d) {
unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * A, * Ap, * Ha, * U, * Q, * H, *B, *C, *Ha_pre_inv;

  /* particular case: */
  if (nb_eqs==0) {
    *Bp = Matrix_Alloc(0, E->NbColumns);
    *Cp = Matrix_Alloc(0, E->NbColumns);
    *d = NULL;
    return NULL;
  }

  /* 1- build A, the part of E corresponding to the variables */
  A = Matrix_Alloc(nb_eqs, nb_vars);
  for (i=0; i< E->NbRows; i++) {
    for (j=0; j< nb_vars; j++)
      value_assign(A->p[i][j],E->p[i][j+1]);
  }
  
  /* 2- Compute Ha^-1, where Ha is the left HNF of A
      a/ Compute H = [Ha 0] */
  left_hermite(A, &H, &Q, &U);
  Matrix_Free(A);
  Matrix_Free(Q);
  
  /*   b/ just keep the m x m matrix Ha */
  Ha = Matrix_Alloc(nb_eqs, nb_eqs);
  for (i=0; i< nb_eqs; i++) {
    for (j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j],H->p[i][j]);
    }
  }
  Matrix_Free(H);

  // show_matrix(Ha);

  /*  c/ Invert Ha */
  Ha_pre_inv = Matrix_Alloc(nb_eqs, nb_eqs+1);
  if(!MatInverse(Ha, Ha_pre_inv)) { 
    fprintf(stderr,"extract_funny_stuff > Matrix Ha is non-invertible.");
  }
  /* store back Ha^-1  in Ha, to spare a MatrixAlloc/MatrixFree */
  for(i=0; i< nb_eqs; i++) {
    for(j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j],Ha_pre_inv->p[i][j]);
    }
  }
  // show_matrix(Ha_pre_inv);
  /* the diagonal elements of D are stored in 
     the last column of Ha_pre_inv (property of MatInverse). */
  (*d) = Matrix_Alloc(Ha_pre_inv->NbRows, 1);

  for (i=0; i< Ha_pre_inv->NbRows; i++) 
    value_assign((*d)->p[i][0], Ha_pre_inv->p[i][Ha_pre_inv->NbColumns-1]);
 
  // show_matrix(Ha);
  // show_matrix(*d);
  Matrix_Free(Ha_pre_inv);

  /* 3- Build B'and C'
      compute B' */
  B = Matrix_Alloc(nb_eqs,nb_parms);
  for(i=0; i< E->NbRows; i++) {
    for(j=0; j< nb_parms; j++)
      value_assign(B->p[i][j], E->p[i][1+nb_vars+j]);
  }

  (*Bp) = Matrix_Alloc(B->NbRows,B->NbColumns);
  // show_matrix(B);
  Matrix_Product(Ha, B, (*Bp));
  // show_matrix(*Bp);
  Matrix_Free(B);
  
  /* compute C' */
  C = Matrix_Alloc(nb_eqs,1);
  for(i=0; i< E->NbRows; i++) {
    value_assign(C->p[i][0], E->p[i][E->NbColumns-1]);
  }

  // show_matrix(C);
  (*Cp) = Matrix_Alloc(nb_eqs, 1);
  Matrix_Product(Ha, C, (*Cp));
  // show_matrix(*Cp);
  Matrix_Free(C);

  Matrix_Free(Ha);
  return U;
}
  

/* 
given a parameterized constraints matrix with m equalities, computes the compression matrix 
 G such that there is an integer solution in the variables space for each value of N', with 
 N = G N' (N are the "nb_parms" parameters) 
*/
Matrix * compress_parms(Matrix * E, int nb_parms) {
  unsigned int i,j, k, nb_eqs=0;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix *U, *d, *Bp, *Cp, *G;

  /* particular case where there is no equation */
  if (E->NbRows==0) return Identity_Matrix(nb_parms+1);

  U = extract_funny_stuff(E, nb_parms, &Bp, & Cp, &d); 

  Matrix_Free(U);
  /* The compression matrix N = G.N' must be such that (B'N+C') mod d = 0 (1) */

  /* the Ns verifying the system B'N+C' = 0 are solutions of (1)
     K is a basis of the integer kernel of B: its column-vectors link two solutions of (1)
     Moreover, B'_iN mod d is periodic of period (s_ik):
     B'N mod d is periodic of period (s_k) = lcm_i(s_ik)
     The linear part of G is given by the HNF of (K | S), 
     where S is the full-dimensional diagonal matrix (s_k)
     the constant part of G is a particular solution of (1)
     if no integer constant part is found, there is no solution. */

  G = int_mod_basis(Bp, Cp, d);
  Matrix_Free(Bp);
  Matrix_Free(Cp);
  Matrix_Free(d);
  return G;
}

/* given a matrix with m parameterized equations, 
 compress the nb_parms parameters and n-m variables so that 
 m variables are intaseger,
 and transform the variable space into a n-m space 
 by eliminating the m variables (using the equalities)
 the variables to be eliminated are chosen automatically by the function
*/
Matrix * full_dimensionize(Matrix const * M, int nb_parms, Matrix ** Validity_Lattice) {
  Matrix * Eqs, * Ineqs;
  Matrix * Permuted_Eqs, * Permuted_Ineqs;
  Matrix * Full_Dim;
  Matrix * WVL; /* The Whole Validity Lattice (vars+parms) */
  unsigned int i,j;
  int nb_elim_vars;
  int * permutation, * permutation_inv;
  /* 0- Split the equalities and inequalities from each other */
  split_constraints(M, &Eqs, &Ineqs);

  /* 1- if the polyhedron is already full-dimensional, return it */
  if (Eqs->NbRows==0) {
    Matrix_Free(Eqs);
    (*Validity_Lattice) = Identity_Matrix(nb_parms+1);
    return Ineqs;
  }
  nb_elim_vars = Eqs->NbRows;

  /* 2- put the vars to be eliminated at the first positions, 
     and compress the other vars/parms
     -> [ variables to eliminate / parameters / variables to keep ] */
  permutation = find_a_permutation(Eqs, nb_parms);
  Permuted_Eqs = mpolyhedron_permute(Eqs, permutation);
  WVL = compress_parms(Permuted_Eqs, Eqs->NbColumns-2-Eqs->NbRows);
  mpolyhedron_compress_last_vars(Permuted_Eqs, WVL);
  Permuted_Ineqs = mpolyhedron_permute(Ineqs, permutation);
  Matrix_Free(Eqs);
  Matrix_Free(Ineqs);
  mpolyhedron_compress_last_vars(Permuted_Ineqs, WVL);

  /* 3- eliminate the first variables */
  if (!mpolyhedron_eliminate_first_variables(Permuted_Eqs, Permuted_Ineqs)) {
    fprintf(stderr,"full-dimensionize > variable elimination failed. \n"); 
    return NULL;
  }
  // show_matrix(Permuted_Eqs);
  // show_matrix(Permuted_Ineqs);

  /* 4- get rid of the first (zero) columns, 
     which are now useless, and put the parameters back at the end */
  Full_Dim = Matrix_Alloc(Permuted_Ineqs->NbRows, Permuted_Ineqs->NbColumns-nb_elim_vars);
  for (i=0; i< Permuted_Ineqs->NbRows; i++) {
    value_set_si(Full_Dim->p[i][0], 1);
    for (j=0; j< nb_parms; j++) 
      value_assign(Full_Dim->p[i][j+Full_Dim->NbColumns-nb_parms-1], 
		   Permuted_Ineqs->p[i][j+nb_elim_vars+1]);
    for (j=0; j< Permuted_Ineqs->NbColumns-nb_parms-2-nb_elim_vars; j++) 
      value_assign(Full_Dim->p[i][j+1], 
		   Permuted_Ineqs->p[i][nb_elim_vars+nb_parms+j+1]);
    value_assign(Full_Dim->p[i][Full_Dim->NbColumns-1], 
		 Permuted_Ineqs->p[i][Permuted_Ineqs->NbColumns-1]);
  }
  Matrix_Free(Permuted_Ineqs);
  
  /* 5- Keep only the the validity lattice restricted to the parameters */
  *Validity_Lattice = Matrix_Alloc(nb_parms+1, nb_parms+1);
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++)
      value_assign((*Validity_Lattice)->p[i][j], 
		   WVL->p[i][j]);
    value_assign((*Validity_Lattice)->p[i][nb_parms], 
		 WVL->p[i][WVL->NbColumns-1]);
  }
  for (j=0; j< nb_parms; j++) 
    value_set_si((*Validity_Lattice)->p[nb_parms][j], 0);
  value_assign((*Validity_Lattice)->p[nb_parms][nb_parms], 
	       WVL->p[WVL->NbColumns-1][WVL->NbColumns-1]);

  /* 6- Clean up */
  Matrix_Free(WVL);
  return Full_Dim;
} /* full_dimensionize */

