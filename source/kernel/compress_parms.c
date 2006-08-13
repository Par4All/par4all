/** 
 * $Id: compress_parms.c,v 1.24 2006/08/13 09:37:39 skimo Exp $
 *
 * The integer points in a parametric linear subspace of Q^n are generally
 * lying on a sub-lattice of Z^n.  To simplify, the functions here compress
 * some parameters so that the variables are integer for any intger values of
 * the parameters.
 * @author B. Meister 12/2003-2006 meister@icps.u-strasbg.fr
 * LSIIT -ICPS 
 * UMR 7005 CNRS
 * Louis Pasteur University (ULP), Strasbourg, France 
*/

#include <stdlib.h>
#include <polylib/polylib.h>

/** 
 * debug flags (2 levels)
 */
#define dbgCompParm 0
#define dbgCompParmMore 0

#define dbgStart(a) if (dbgCompParmMore) { printf(" -- begin "); \
                                           printf(#a);        \
					   printf(" --\n"); }   \
                                           while(0)
#define dbgEnd(a) if (dbgCompParmMore) { printf(" -- end "); \
                                         printf(#a);      \
					 printf(" --\n"); } \
                                         while(0)

/** 
 * Given a full-row-rank nxm matrix M made of m row-vectors), computes the
 * basis K (made of n-m column-vectors) of the integer kernel of the rows of M
 * so we have: M.K = 0
*/
Matrix * int_ker(Matrix * M) {
  Matrix *U, *Q, *H, *H2, *K;
  int i, j, rk;

  if (dbgCompParm)
    show_matrix(M);
  /* eliminate redundant rows : UM = H*/
  right_hermite(M, &H, &Q, &U);
  for (rk=H->NbRows-1; (rk>=0) && Vector_IsZero(H->p[rk], H->NbColumns); rk--);
  rk++;
  if (dbgCompParmMore) {
    printf("rank = %d\n", rk);
  }
    
  /* there is a non-null kernel if and only if the dimension m of 
     the space spanned by the rows 
     is inferior to the number n of variables */
  if (M->NbColumns <= rk) {
    Matrix_Free(H);
    Matrix_Free(Q);
    Matrix_Free(U);
    K = Matrix_Alloc(M->NbColumns, 0);
    return K;
  }
  Matrix_Free(U); 
  Matrix_Free(Q);
  /* fool left_hermite  by giving NbRows =rank of M*/
  H->NbRows=rk;
  /* computes MU = [H 0] */
  left_hermite(H, &H2, &Q, &U); 
   if (dbgCompParmMore) {
    printf("-- Int. Kernel -- \n");
    show_matrix(M);
    printf(" = \n");
    show_matrix(H2);
    show_matrix(U); 
  }
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
} /* int_ker */


/** 
 * Computes the intersection of two linear lattices, whose base vectors are
 * respectively represented in A and B.
 * <p>
 * Temporary pre-condition: A and B must represent full-dimensional lattices of
 * the same dimension.
 * </p>
 * If I and/or Lb is set to NULL, then the matrix is allocated. 
 * Else, the matrix is assumed to be allocated already. 
 * I and Lb are rk x rk, where rk is the rank of A (or B).
 * @param A the matrix whose column-vectors are the basis for the first linear
 * lattice.
 * @param B the matrix whose column-vectors are the basis for the second linear
 * lattice.
 * @param Lb the matrix such that B.Lb = I, where I is the intersection.
 * @return their intersection.
 */
static void linearInter(Matrix * A, Matrix * B, Matrix ** I, Matrix **Lb) {
  Matrix * AB=NULL;
  int rk = A->NbRows;
  int i,j;

  Matrix * H, *U, *Q;
  assert(A->NbColumns == rk && B->NbRows==rk && B->NbColumns == rk);
  /* 1- build the matrix 
   * (A 0 1)
   * (0 B 1)
   */
  AB = Matrix_Alloc(2*rk, A->NbColumns+B->NbColumns+rk);
  for (i=0; i< rk; i++) {
    for (j=0; j<rk; j++) {
      value_assign(AB->p[i][j], A->p[i][j]);
    }
  }
  for (i=0; i< rk; i++) {
    for (j=0; j<rk; j++) {
      value_assign(AB->p[i+rk][j+rk], B->p[i][j]);
    }
  }
  for (i=0; i< rk; i++) {
      value_set_si(AB->p[i][i+2*rk], 1);
      value_set_si(AB->p[i+rk][i+2*rk], 1);
  }
  if (dbgCompParm) {
    show_matrix(AB);
  }

  /* 2- Compute its left Hermite normal form. AB.U = [H 0] */
  left_hermite(AB, &H, &Q, &U);
  Matrix_Free(AB);
  Matrix_Free(H);
  Matrix_Free(Q);

  /* if you split U evenly in 9 submatrices, you have: 
   * A.U_13 = -U_33
   * B.U_23 = -U_33 
   * U_33 is a (the smallest) combination of col-vectors of A and B at the same
   * time: their intersection.
  */
  Matrix_subMatrix(U, 2*rk, 2*rk, rk, rk, I);
  Matrix_oppose(*I);
  Matrix_subMatrix(U, rk, 2*rk, rk, rk, Lb);
  Matrix_oppose(*Lb);
  if (dbgCompParm) {
    show_matrix(U);
  }
  Matrix_Free(U);
} /* linearInter */


/** 
 * Given a system of equalities, looks if it has an integer solution in the
 * combined space, and if yes, returns one solution.
 * <p>pre-condition: the equalities are full-row rank (without the constant
 * part)</p>
 * @param Eqs the system of equations (as constraints)
 * @return NULL a feasible integer solution if it exists, else NULL.
 */
static Matrix * integerSolution(Matrix * Eqs) {
  Matrix * Hm, *H=NULL, *U, *Q, *M=NULL, *C=NULL, *Hi;
  Matrix * I, *Ip;
  int i;
  Value mod;
  if (Eqs==NULL) return NULL;
  /* we use: AI = C = (Ha 0).Q.I = (Ha 0)(I' 0)^T */
  /* with I = Qinv.I' = U.I'*/
  /* 1- compute I' = Hainv.C */
  /* HYP: the equalities are full-row rank */
  unsigned int rk = Eqs->NbRows;
  Matrix_subMatrix(Eqs, 0, 1, rk, Eqs->NbColumns-2, &M);
  left_hermite(M, &Hm, &Q, &U);
  Matrix_subMatrix(Hm, 0,0, rk,rk, &H);
  if (dbgCompParmMore) {
    show_matrix(Hm);
    show_matrix(H);
    show_matrix(U);
  }
  Matrix_Free(Q);
  Matrix_Free(Hm);
  Matrix_subMatrix(Eqs, 0, Eqs->NbColumns-1, rk, 1, &C);
  Matrix_oppose(C);
  Hi = Matrix_Alloc(rk, rk+1);
  MatInverse(H, Hi);
  if (dbgCompParmMore) {
    show_matrix(C);
    show_matrix(Hi);
  }
  /* put the numerator of Hinv back into H */
  Matrix_subMatrix(Hi, 0, 0, rk, rk, &H);
  Ip = Matrix_Alloc(Eqs->NbColumns-2, 1);
  /* fool Matrix_Product on the size of Ip */
  Ip->NbRows = rk;
  Matrix_Product(H, C, Ip);
  Ip->NbRows = Eqs->NbColumns-2;
  Matrix_Free(H);
  Matrix_Free(C);
  value_init(mod);
  for (i=0; i< rk; i++) {
    /* if Hinv.C is not integer, return NULL (no solution) */
    value_pmodulus(mod, Ip->p[i][0], Hi->p[i][rk]);
    if (value_notzero_p(mod)) { 
      return NULL;
    }
    else {
      value_pdivision(Ip->p[i][0], Ip->p[i][0], Hi->p[i][rk]);
    }
  }
  /* fill the rest of I' with zeros */
  for (i=rk; i< Eqs->NbColumns-2; i++) {
    value_set_si(Ip->p[i][0], 0);
  }
  value_clear(mod);
  Matrix_Free(Hi);
  /* 2 - Compute the particular solution I = U.(I' 0) */
  I = Matrix_Alloc(Eqs->NbColumns-2, 1);
  Matrix_Product(U, Ip, I);
  if (dbgCompParm) {
    show_matrix(I);
  }
  return I;
}


/** 
 * Returns the smallest linear compression of the parameters necessary for an
 * integer value of the variables to exist for each integer value of the
 * parameters.
 */
static Matrix * minLinearCompress(Matrix * A, Matrix * B, unsigned int nbParms) {
  Matrix * Lb= NULL;
  Matrix * I = NULL;
  Matrix * Hb = NULL;
  Matrix * Hg = NULL;
  Matrix *U, *H, *Q;
  int rk = A->NbRows;
  assert(B->NbColumns==nbParms);
  if (nbParms==0) {
    Matrix_identity(rk,&Hg);
    return Hg;
  }
  /* HYP: A is square, full-rank. */
  /* HYP: B is full row-rank. */
  left_hermite(B, &H, &Q, &U);
  Matrix_subMatrix(H, 0, 0, rk, rk, &Hb);
  Matrix_Free(H);
  linearInter(A, Hb, &I, &Lb);
  if (dbgCompParm) {
    show_matrix(I);
    show_matrix(Lb);
  }
  Matrix_Free(I);
  Matrix_Free(Hb);
  // HYP: A is square, full-rank.
  // Note: we juste reuse U, which has the appropriate dimensions
  Matrix_identity(nbParms, &Hg);
  Matrix_copySubMatrix(Lb, 0, 0, rk, rk, Hg, 0,0);
  /* the linear part of the minimal lattice is U.Hg */
  Matrix_Product(U, Hg, Q);
  return Q;
}/* minLinearCompress */


/**
 * Finds the smallest affine compression of the parameters necessary for an
 * integer value of the variables to exist for each integer value of the
 * parameters.
 * @return 0 if there is no solution, 1 if there is one 
 * (and then this compression lattice is in G)
 */
int minAffineCompress(Matrix * Eqs, unsigned int nbParms, 
			      Matrix ** G) {
  int nbVars = Eqs->NbColumns - nbParms -2;
  Matrix * N0 = integerSolution(Eqs);
  Matrix * A = NULL;
  Matrix * B = NULL;
  Matrix * G0=NULL;
  // if there is no integer solution, return false
  if (N0==NULL) {
    return 0;
  }
  else {
    if ((*G)==NULL) {
      (*G) = Matrix_Alloc(nbParms+1, nbParms+1);
    }
    else {
      assert((*G)->NbRows==nbParms+1 && (*G)->NbColumns==nbParms+1);
    }
    Matrix_copySubMatrix(N0, nbVars, 0, nbParms, 1, (*G), 0, nbParms);
    value_set_si((*G)->p[nbParms][nbParms], 1);
    Matrix_Free(N0);
    Matrix_subMatrix(Eqs, 0, 1, Eqs->NbRows, nbVars, &A);
    Matrix_subMatrix(Eqs, 0, nbVars+1, Eqs->NbRows, nbParms, &B);
    //HYP: A is square and full-rank
    G0 = minLinearCompress(A, B, nbParms);
    Matrix_Free(A);
    Matrix_Free(B);
    Matrix_copySubMatrix(G0, 0, 0, nbParms, nbParms, (*G), 0, 0);
    return 1;
  }
}/* minAffineCompress */


/**
 * Eliminate the columns corresponding to the eliminated parameters.
 * @param M the constraints matrix whose columns are to be removed
 * @param nbVars an offset to be added to the ranks of the variables to be
 * removed
 * @param elimParms the list of ranks of the variables to be removed
 * @param newM (output) the matrix without the removed columns
 */
void Constraints_removeElimCols(Matrix * M, unsigned int nbVars, 
			   unsigned int *elimParms, Matrix ** newM) {
  unsigned int i, j, k;
  if (elimParms[0]==0) {
    Matrix_clone(M, newM);
    return;
  }
  if ((*newM)==NULL) {
    (*newM) = Matrix_Alloc(M->NbRows, M->NbColumns - elimParms[0]);
  }
  else {
    assert ((*newM)->NbColumns==M->NbColumns - elimParms[0]);
  }
  for (i=0; i< M->NbRows; i++) {
    value_assign((*newM)->p[i][0], M->p[i][0]); /* kind of cstr */
    k=0;
    Vector_Copy(&(M->p[i][1]), &((*newM)->p[i][1]), nbVars);
    for (j=0; j< M->NbColumns-2-nbVars; j++) {
      if (j!=elimParms[k+1]) {
	value_assign((*newM)->p[i][j-k+nbVars+1], M->p[i][j+nbVars+1]);
      }
      else {
	k++;
      }
    }
    value_assign((*newM)->p[i][(*newM)->NbColumns-1], 
		 M->p[i][M->NbColumns-1]); /* cst part */
  }
} /* Constraints_removeElimCols */


/**
 * Eliminates all the equalities in a set of constraints and returns the set of
 * constraints defining a full-dimensional polyhedron, such that there is a
 * bijection between integer points of the original polyhedron and these of the
 * resulting (projected) polyhedron).
 * If VL is set to NULL, this funciton allocates it. Else, it assumes that
 * (*VL) points to a matrix of the right size.
 * <p> The following things are done: 
 * <ol>
 * <li> remove equalities involving only parameters, and remove as many
 *      parameters as there are such equalities. From that, the list of
 *      eliminated parameters <i>elimParms</i> is built.
 * <li> remove equalities that involve variables. This requires a compression
 *      of the parameters and of the other variables that are not eliminated.
 *      The affine compresson is represented by matrix VL (for <i>validity
 *      lattice</i>) and is such that (N I 1)^T = VL.(N' I' 1), where N', I'
 *      are integer (they are the parameters and variables after compression).
 *</ol>
 *</p>
 */
void Constraints_fullDimensionize(Matrix ** M, Matrix ** C, Matrix ** VL, 
				  Matrix ** Eqs, Matrix ** ParmEqs, 
				  unsigned int ** elimVars, 
				  unsigned int ** elimParms,
				  int maxRays) {
  unsigned int i, j;
  Matrix * A=NULL, *B=NULL;
  Matrix * Ineqs=NULL;
  unsigned int nbVars = (*M)->NbColumns - (*C)->NbColumns;
  unsigned int nbParms;
  int nbElimVars;
  Matrix * fullDim = NULL;

  /* variables for permutations */
  unsigned int * permutation, * permutationInv;
  Matrix * permutedEqs=NULL, * permutedIneqs=NULL;
  
  /* 1- Eliminate the equalities involving only parameters. */
  (*ParmEqs) = Constraints_removeParmEqs(M, C, 0, elimParms);
  /* if the polyehdron is empty, return now. */
  if ((*M)->NbColumns==0) return;
  /* eliminate the columns corresponding to the eliminated parameters */
  if (elimParms[0]!=0) {
    Constraints_removeElimCols(*M, nbVars, (*elimParms), &A);
    Matrix_Free(*M);
    (*M) = A;
    Constraints_removeElimCols(*C, 0, (*elimParms), &B);
    Matrix_Free(*C);
    (*C) = B;
    if (dbgCompParm) {
      printf("After false parameter elimination: \n");
      show_matrix(*M);
      show_matrix(*C);
    }
  }
  nbParms = (*C)->NbColumns-2;

  /* 2- Eliminate the equalities involving variables */
  /*   a- extract the (remaining) equalities from the poyhedron */
  split_constraints((*M), Eqs, &Ineqs);
  nbElimVars = (*Eqs)->NbRows;
  /*    if the polyhedron is already full-dimensional, return */
  if ((*Eqs)->NbRows==0) {
    Matrix_identity(nbParms+1, VL);
    return;
  }
  /*   b- choose variables to be eliminated */
  permutation = find_a_permutation((*Eqs), nbParms);

  if (dbgCompParm) {
    printf("Permuting the vars/parms this way: [ ");
    for (i=0; i< (*Eqs)->NbColumns-2; i++) {
      printf("%d ", permutation[i]);
    }
    printf("]\n");
  }

  Constraints_permute((*Eqs), permutation, &permutedEqs);
  minAffineCompress(permutedEqs, (*Eqs)->NbColumns-2-(*Eqs)->NbRows, VL);

  if (dbgCompParm) {
    printf("Validity lattice: ");
    show_matrix(*VL);
  }
  Constraints_compressLastVars(permutedEqs, (*VL));
  Constraints_permute(Ineqs, permutation, &permutedIneqs);
  if (dbgCompParmMore) {
    show_matrix(permutedIneqs);
    show_matrix(permutedEqs);
  }
  Matrix_Free(*Eqs);
  Matrix_Free(Ineqs);
  Constraints_compressLastVars(permutedIneqs, (*VL));
  if (dbgCompParm) {
    printf("After compression: ");
    show_matrix(permutedIneqs);
  }
  /*   c- eliminate the first variables */
  assert(Constraints_eliminateFirstVars(permutedEqs, permutedIneqs));
  if (dbgCompParmMore) {
    printf("After elimination of the variables: ");
    show_matrix(permutedIneqs);
  }

  /*   d- get rid of the first (zero) columns, 
       which are now useless, and put the parameters back at the end */
  fullDim = Matrix_Alloc(permutedIneqs->NbRows,
			 permutedIneqs->NbColumns-nbElimVars);
  for (i=0; i< permutedIneqs->NbRows; i++) {
    value_set_si(fullDim->p[i][0], 1);
    for (j=0; j< nbParms; j++) {
      value_assign(fullDim->p[i][j+fullDim->NbColumns-nbParms-1], 
		   permutedIneqs->p[i][j+nbElimVars+1]);
    }
    for (j=0; j< permutedIneqs->NbColumns-nbParms-2-nbElimVars; j++) {
      value_assign(fullDim->p[i][j+1], 
		   permutedIneqs->p[i][nbElimVars+nbParms+j+1]);
    }
    value_assign(fullDim->p[i][fullDim->NbColumns-1], 
		 permutedIneqs->p[i][permutedIneqs->NbColumns-1]);
  }
  Matrix_Free(permutedIneqs);

} /* Constraints_fullDimensionize */


/**
 * Given a matrix that defines a full-dimensional affine lattice, returns the 
 * affine sub-lattice spanned in the k first dimensions.
 * Useful for instance when you only look for the parameters' validity lattice.
 * @param lat the original full-dimensional lattice
 * @param subLat the sublattice
 */
void Matrix_extractSubLattice(Matrix * lat, unsigned int k, Matrix ** subLat) {
  Matrix * H, *Q, *U, *linLat = NULL;
  unsigned int i;
  dbgStart(Matrix_extractSubLattice);
  /* if the dimension is already good, just copy the initial lattice */
  if (k==lat->NbRows-1) {
    if (*subLat==NULL) {
      (*subLat) = Matrix_Copy(lat);
    }
    else {
      Matrix_copySubMatrix(lat, 0, 0, lat->NbRows, lat->NbColumns, (*subLat), 0, 0);
    }
    return;
  }
  assert(k<lat->NbRows-1);
  /* 1- Make the linear part of the lattice triangular to eliminate terms from 
     other dimensions */
  Matrix_subMatrix(lat, 0, 0, lat->NbRows, lat->NbColumns-1, &linLat);
  // OPT: any integer column-vector elimination is ok indeed.
  // OPT: could test if the lattice is already in triangular form.
  left_hermite(linLat, &H, &Q, &U);
  if (dbgCompParmMore) {
    show_matrix(H);
  }
  Matrix_Free(Q);
  Matrix_Free(U);
  Matrix_Free(linLat);
  /* if not allocated yet, allocate it */
  if (*subLat==NULL) {
    (*subLat) = Matrix_Alloc(k+1, k+1);
  }
  Matrix_copySubMatrix(H, 0, 0, k, k, (*subLat), 0, 0);
  Matrix_Free(H);
  Matrix_copySubMatrix(lat, 0, lat->NbColumns-1, k, 1, (*subLat), 0, k);
  for (i=0; i<k; i++) {
    value_set_si((*subLat)->p[k][i], 0);
  }
  value_set_si((*subLat)->p[k][k], 1);
  dbgEnd(Matrix_extractSubLattice);
} /* Matrix_extractSubLattice */


/** 
 * Computes the overall period of the variables I for (MI) mod |d|, where M is
 * a matrix and |d| a vector. Produce a diagonal matrix S = (s_k) where s_k is
 * the overall period of i_k 
 * @param M the set of affine functions of I (row-vectors)
 * @param d the column-vector representing the modulos
*/
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
} /* affine_periods */


/** 
 * Given a matrix B' with m rows and m-vectors C' and d, computes the basis of
 * the integer solutions to (B'N+C') mod d = 0 (1).  the Ns verifying the
 * system B'N+C' = 0 are solutions of (1) K is a basis of the integer kernel of
 * B: its column-vectors link two solutions of (1) <p>
 * Moreover, B'_iN mod d is periodic of period (s_ik): B'N mod d is periodic of
 * period (s_k) = lcm_i(s_ik). 
 * The linear part of G is given by the HNF of (K | S), where S is the
 * full-dimensional diagonal matrix (s_k) the constant part of G is a
 * particular solution of (1) if no integer constant part is found, there is no
 * solution and this function returns NULL.
*/
Matrix * int_mod_basis(Matrix * Bp, Matrix * Cp, Matrix * d) {
  int nb_eqs = Bp->NbRows;
  unsigned int nb_parms=Bp->NbColumns;
  unsigned int i, j;
  Matrix *H, *U, *Q, *M, *inv_H_M, *Ha, *Np_0, *N_0, *G, *K, *S, *KS;
  Value tmp;

  value_init(tmp);
  /*   a/ compute K and S */
  /* simplify the constraints */
  /* for (i=0; i< Bp->NbRows; i++)
    for (j=0; j< Bp->NbColumns; j++) 
      value_pmodulus(Bp->p[i][j], Bp->p[i][j], d->p[0][i]);
  */
  K = int_ker(Bp);
  S = affine_periods(Bp, d);
  if (dbgCompParmMore) {
    show_matrix(K); 
    show_matrix(S);
  }
  
  /*   b/ compute the linear part of G : HNF(K|S) */

  /* fill K|S */
  KS = Matrix_Alloc(nb_parms, K->NbColumns+ nb_parms);
  for(i=0; i< KS->NbRows; i++) {
    for(j=0; j< K->NbColumns; j++) value_assign(KS->p[i][j], K->p[i][j]);
    for(j=0; j< S->NbColumns; j++) value_assign(KS->p[i][j+K->NbColumns],
						S->p[i][j]);
  }
  Matrix_Free(K);

  if (dbgCompParmMore) { 
    show_matrix(KS);
  }

  /* HNF(K|S) */
  left_hermite(KS, &H, &U, &Q);
  Matrix_Free(KS);
  Matrix_Free(U);
  Matrix_Free(Q);
  
  if (dbgCompParm) {
    printf("HNF(K|S) = ");
    show_matrix(H);
  }

  /* put HNF(K|S) in the p x p matrix S (which has already the appropriate size
     so we save a Matrix_Alloc) */
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++) {
      value_assign(S->p[i][j], H->p[i][j]);
    }
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
  Matrix_Free(M);
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
    value_modulus(tmp, Np_0->p[i][0], inv_H_M->p[i][nb_eqs]);
    if (value_zero_p(tmp))
      value_division(Np_0->p[i][0], Np_0->p[i][0], inv_H_M->p[i][nb_eqs]);
    else {
      value_clear(tmp);
      Matrix_Free(S);
      Matrix_Free(inv_H_M);
      Matrix_Free(Np_0);
      fprintf(stderr, "int_mod_basis > "
              "No particular solution: polyhedron without integer points.\n");
      return NULL;
    }
  }
  Matrix_Free(inv_H_M);
  if (dbgCompParmMore) {
    show_matrix(Np_0);
  }

  /* now compute the actual particular solution N_0 = U_M. N'_0 */
  N_0 = Matrix_Alloc(U->NbColumns, 1);
  /* OPT: seules les nb_eq premières valeurs de N_0 sont utiles en fait. */
  Matrix_Product(U, Np_0, N_0);
  
  if (dbgCompParm) {
    show_matrix(N_0);
  }
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
  value_clear(tmp);
  Matrix_Free(S);
  Matrix_Free(N_0);
  return G;
} /* int_mod_basis */


/** 
 * Utility function: given a matrix containing the equations AI+BN+C=0,
 * computes the HNF of A : A = [Ha 0].Q and return :
 * <ul>
 * <li> B'= H^-1.(-B) 
 * <li> C'= H^-1.(-C)
 * <li> U = Q^-1 (-> return value)
 * <li> D, 
 * </ul>
 * where Ha^-1 = D^-1.H^-1 with H and D integer matrices. 
 * In fact, as D is diagonal, we return a column-vector d.
 * Note: ignores the equalities that involve only parameters
*/
static Matrix * extract_funny_stuff(Matrix * const E, int nb_parms, 
			     Matrix ** Bp, Matrix **Cp, Matrix **d) {
unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * A, * Ap, * Ha, * U, * Q, * H, *B, *C, *Ha_pre_inv;

  /* Only deal with the equalities involving variables:
     - don't count them
     - mark them (a 2 instead of the 0 in 1st column
  */
  for (i=0; i< E->NbRows; i++) {
    if (First_Non_Zero(E->p[i], E->NbColumns)>= nb_vars+1) {
      value_set_si(E->p[i][0], 2);
      nb_eqs--;
    }
  }

  /* particular case: 
    - no equality (of interest) in E */
  if (nb_eqs==0) {
    *Bp = Matrix_Alloc(0, E->NbColumns);
    *Cp = Matrix_Alloc(0, E->NbColumns);
    *d = NULL;
    /* unmark the equalities that we filtered out */
    for (i=0; i< E->NbRows; i++) {
      value_set_si(E->p[i][0], 0);
    }
    return NULL;
  }
    
  /* 1- build A, the part of E corresponding to the variables */
  A = Matrix_Alloc(nb_eqs, nb_vars);
  for (i=0; i< E->NbRows; i++) {
    if (value_zero_p(E->p[i][0])) {
      for (j=0; j< nb_vars; j++) {
	value_assign(A->p[i][j],E->p[i][j+1]);
      }
    }
  }
  if (dbgCompParmMore) {
    show_matrix(A);
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


  /*  c/ Invert Ha */
  Ha_pre_inv = Matrix_Alloc(nb_eqs, nb_eqs+1);
  assert (MatInverse(Ha, Ha_pre_inv));
  if (dbgCompParmMore) {
    show_matrix(Ha_pre_inv);
  }

  /* store back Ha^-1  in Ha, to save a MatrixAlloc/MatrixFree */
  for(i=0; i< nb_eqs; i++) {
    for(j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j],Ha_pre_inv->p[i][j]);
    }
  }

  /* the diagonal elements of D are stored in 
     the last column of Ha_pre_inv (property of MatInverse). */
  (*d) = Matrix_Alloc(Ha_pre_inv->NbRows, 1);

  for (i=0; i< Ha_pre_inv->NbRows; i++) {
    value_assign((*d)->p[i][0], Ha_pre_inv->p[i][Ha_pre_inv->NbColumns-1]);
  }
 
  Matrix_Free(Ha_pre_inv);

  /* 3- Build B'and C'
      compute B' */
  B = Matrix_Alloc(nb_eqs,nb_parms);
  for(i=0; i< E->NbRows; i++) {
    if (value_zero_p(E->p[i][0])) {
      for(j=0; j< nb_parms; j++) {
	value_assign(B->p[i][j], E->p[i][1+nb_vars+j]);
      }
    }
  }
  

  (*Bp) = Matrix_Alloc(B->NbRows,B->NbColumns);
  Matrix_Product(Ha, B, (*Bp));
  Matrix_Free(B);
  
  /* compute C' */
  C = Matrix_Alloc(nb_eqs,1);
  for(i=0; i< E->NbRows; i++) {
    if (value_zero_p(E->p[i][0])) {
      value_assign(C->p[i][0], E->p[i][E->NbColumns-1]);
    }
  }
  
  /* unmark the equalities that we filtered out */
  for (i=0; i< E->NbRows; i++) {
    value_set_si(E->p[i][0], 0);
  }
  
  (*Cp) = Matrix_Alloc(nb_eqs, 1);
  Matrix_Product(Ha, C, (*Cp));
  Matrix_Free(C);

  Matrix_Free(Ha);
  return U;
} /* extract_funny_stuff */
  

/** 
 * Given a parameterized constraints matrix with m equalities, computes the
 * compression matrix G such that there is an integer solution in the variables
 * space for each value of N', with N = G N' (N are the "nb_parms" parameters)
 * @param E a matrix of parametric equalities @param nb_parms the number of
 * parameters
*/
Matrix * compress_parms(Matrix * E, int nb_parms) {
  unsigned int i,j, k, nb_eqs=0;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix *U, *d, *Bp, *Cp, *G;

  /* particular case where there is no equation */
  if (E->NbRows==0) return Identity_Matrix(nb_parms+1);

  U = extract_funny_stuff(E, nb_parms, &Bp, & Cp, &d); 
  if (dbgCompParmMore) {
    show_matrix(Bp); 
    show_matrix(Cp);
    show_matrix(d);
  }

  Matrix_Free(U);
  /* The compression matrix N = G.N' must be such that (B'N+C') mod d = 0 (1)
  */

  /* the Ns verifying the system B'N+C' = 0 are solutions of (1) K is a basis
     of the integer kernel of B: its column-vectors link two solutions of (1)
     Moreover, B'_iN mod d is periodic of period (s_ik): B'N mod d is periodic
     of period (s_k) = lcm_i(s_ik) The linear part of G is given by the HNF of
     (K | S), where S is the full-dimensional diagonal matrix (s_k) the
     constant part of G is a particular solution of (1) if no integer constant
     part is found, there is no solution. */

  G = int_mod_basis(Bp, Cp, d);
  Matrix_Free(Bp);
  Matrix_Free(Cp);
  Matrix_Free(d);
  return G;
}/* compress_parms */


/** Removes the equalities that involve only parameters, by eliminating some
 * parameters in the polyhedron's constraints and in the context.<p> 
 * <b>Updates M and Ctxt.</b>
 * @param M1 the polyhedron's constraints
 * @param Ctxt1 the constraints of the polyhedron's context
 * @param renderSpace tells if the returned equalities must be expressed in the
 * parameters space (renderSpace=0) or in the combined var/parms space
 * (renderSpace = 1)
 * @param elimParms the list of parameters that have been removed: an array
 * whose 1st element is the number of elements in the list.  (returned)
 * @return the system of equalities that involve only parameters.
 */
Matrix * Constraints_Remove_parm_eqs(Matrix ** M1, Matrix ** Ctxt1, 
				     int renderSpace, 
				     unsigned int ** elimParms) {
  int i, j, k, nbEqsParms =0;
  int nbEqsM, nbEqsCtxt, allZeros, nbTautoM = 0, nbTautoCtxt = 0;
  Matrix * M = (*M1);
  Matrix * Ctxt = (*Ctxt1);
  int nbVars = M->NbColumns-Ctxt->NbColumns;
  Matrix * Eqs;
  Matrix * EqsMTmp;
  
  /* 1- build the equality matrix(ces) */
  nbEqsM = 0;
  for (i=0; i< M->NbRows; i++) {
    k = First_Non_Zero(M->p[i], M->NbColumns);
    /* if it is a tautology, count it as such */
    if (k==-1) {
      nbTautoM++;
    }
    else {
      /* if it only involves parameters, count it */
      if (k>= nbVars+1) nbEqsM++;
    }
  }

  nbEqsCtxt = 0;
  for (i=0; i< Ctxt->NbRows; i++) {
    if (value_zero_p(Ctxt->p[i][0])) {
      if (First_Non_Zero(Ctxt->p[i], Ctxt->NbColumns)==-1) {
	nbTautoCtxt++;
      }
      else {
	nbEqsCtxt ++;
      }
    }
  }
  nbEqsParms = nbEqsM + nbEqsCtxt; 

  /* nothing to do in this case */
  if (nbEqsParms+nbTautoM+nbTautoCtxt==0) {
    (*elimParms) = (unsigned int*) malloc(sizeof(int));
    (*elimParms)[0] = 0;
    if (renderSpace==0) {
      return Matrix_Alloc(0,Ctxt->NbColumns);
    }
    else {
      return Matrix_Alloc(0,M->NbColumns);
    }
  }
  
  Eqs= Matrix_Alloc(nbEqsParms, Ctxt->NbColumns);
  EqsMTmp= Matrix_Alloc(nbEqsParms, M->NbColumns);
  
  /* copy equalities from the context */
  k = 0;
  for (i=0; i< Ctxt->NbRows; i++) {
    if (value_zero_p(Ctxt->p[i][0]) 
		     && First_Non_Zero(Ctxt->p[i], Ctxt->NbColumns)!=-1) {
      Vector_Copy(Ctxt->p[i], Eqs->p[k], Ctxt->NbColumns);
      Vector_Copy(Ctxt->p[i]+1, EqsMTmp->p[k]+nbVars+1, 
		  Ctxt->NbColumns-1);
      k++;
    }
  }
  for (i=0; i< M->NbRows; i++) {
    j=First_Non_Zero(M->p[i], M->NbColumns);
    /* copy equalities that involve only parameters from M */
    if (j>=nbVars+1) {
      Vector_Copy(M->p[i]+nbVars+1, Eqs->p[k]+1, Ctxt->NbColumns-1);
      Vector_Copy(M->p[i]+nbVars+1, EqsMTmp->p[k]+nbVars+1, 
		  Ctxt->NbColumns-1);
      /* mark these equalities for removal */
      value_set_si(M->p[i][0], 2);
      k++;
    }
    /* mark the all-zero equalities for removal */
    if (j==-1) {
      value_set_si(M->p[i][0], 2);
    }
  }

  /* 2- eliminate parameters until all equalities are used or until we find a
  contradiction (overconstrained system) */
  (*elimParms) = (unsigned int *) malloc((Eqs->NbRows+1) * sizeof(int));
  (*elimParms)[0] = 0;
  allZeros = 0;
  for (i=0; i< Eqs->NbRows; i++) {
    /* find a variable that can be eliminated */
    k = First_Non_Zero(Eqs->p[i], Eqs->NbColumns);
    if (k!=-1) { /* nothing special to do for tautologies */

      /* if there is a contradiction, return empty matrices */
      if (k==Eqs->NbColumns-1) {
	printf("Contradiction in %dth row of Eqs: ",k);
	show_matrix(Eqs);
	Matrix_Free(Eqs);
	Matrix_Free(EqsMTmp);
	(*M1) = Matrix_Alloc(0, M->NbColumns);
	Matrix_Free(M);
	(*Ctxt1) = Matrix_Alloc(0,Ctxt->NbColumns);
	Matrix_Free(Ctxt);
	free(*elimParms);
	(*elimParms) = (unsigned int *) malloc(sizeof(int));
	(*elimParms)[0] = 0;
	if (renderSpace==1) {
	  return Matrix_Alloc(0,(*M1)->NbColumns);
	}
	else {
	  return Matrix_Alloc(0,(*Ctxt1)->NbColumns);
	}
      }	
      /* if we have something we can eliminate, do it in 3 places:
	 Eqs, Ctxt, and M */
      else {
	k--; /* k is the rank of the variable, now */
	(*elimParms)[0]++;
	(*elimParms)[(*elimParms[0])]=k;
	for (j=0; j< Eqs->NbRows; j++) {
	  if (i!=j) {
	    eliminate_var_with_constr(Eqs, i, Eqs, j, k);
	    eliminate_var_with_constr(EqsMTmp, i, EqsMTmp, j, k+nbVars);
	  }
	}
	for (j=0; j< Ctxt->NbRows; j++) {
	  if (value_notzero_p(Ctxt->p[i][0])) {
	    eliminate_var_with_constr(Eqs, i, Ctxt, j, k);
	  }
	}
	for (j=0; j< M->NbRows; j++) {
	  if (value_cmp_si(M->p[i][0], 2)) {
	    eliminate_var_with_constr(EqsMTmp, i, M, j, k+nbVars);
	  }
	}
      }
    }
    /* if (k==-1): count the tautologies in Eqs to remove them later */
    else {
      allZeros++;
    }
  }
  
  /* elimParms may have been overallocated. Now we know how many parms have
     been eliminated so we can reallocate the right amount of memory. */
  if (!realloc((*elimParms), ((*elimParms)[0]+1)*sizeof(int))) {
    fprintf(stderr, "Constraints_Remove_parm_eqs > cannot realloc()");
  }

  Matrix_Free(EqsMTmp);

  /* 3- remove the "bad" equalities from the input matrices
     and copy the equalities involving only parameters */
  EqsMTmp = Matrix_Alloc(M->NbRows-nbEqsM-nbTautoM, M->NbColumns);
  k=0;
  for (i=0; i< M->NbRows; i++) {
    if (value_cmp_si(M->p[i][0], 2)) {
      Vector_Copy(M->p[i], EqsMTmp->p[k], M->NbColumns);
      k++;
    }
  }
  Matrix_Free(M);
  (*M1) = EqsMTmp;
  
  EqsMTmp = Matrix_Alloc(Ctxt->NbRows-nbEqsCtxt-nbTautoCtxt, Ctxt->NbColumns);
  k=0;
  for (i=0; i< Ctxt->NbRows; i++) {
    if (value_notzero_p(Ctxt->p[i][0])) {
      Vector_Copy(Ctxt->p[i], EqsMTmp->p[k], Ctxt->NbColumns);
      k++;
    }
  }
  Matrix_Free(Ctxt);
  (*Ctxt1) = EqsMTmp;
  
  if (renderSpace==0) {// renderSpace = 0: equalities in the parameter space
    EqsMTmp = Matrix_Alloc(Eqs->NbRows-allZeros, Eqs->NbColumns);
    k=0;
    for (i=0; i<Eqs->NbRows; i++) {
      if (First_Non_Zero(Eqs->p[i], Eqs->NbColumns)!=-1) {
	Vector_Copy(Eqs->p[i], EqsMTmp->p[k], Eqs->NbColumns);
	k++;
      }
    }
  }
  else {// renderSpace = 1: equalities rendered in the combined space
    EqsMTmp = Matrix_Alloc(Eqs->NbRows-allZeros, (*M1)->NbColumns);
    k=0;
    for (i=0; i<Eqs->NbRows; i++) {
      if (First_Non_Zero(Eqs->p[i], Eqs->NbColumns)!=-1) {
	Vector_Copy(Eqs->p[i], &(EqsMTmp->p[k][nbVars]), Eqs->NbColumns);
	k++;
      }
    }
  }
  Matrix_Free(Eqs);
  Eqs = EqsMTmp;

  return Eqs;
} /* Constraints_Remove_parm_eqs */


/** Removes equalities involving only parameters, but starting from a
 * Polyhedron and its context.
 * @param P the polyhedron
 * @param C P's context
 * @param renderSpace: 0 for the parameter space, =1 for the combined space.
 * @maxRays Polylib's usual <i>workspace</i>.
 */
Polyhedron * Polyhedron_Remove_parm_eqs(Polyhedron ** P, Polyhedron ** C, 
					int renderSpace, 
					unsigned int ** elimParms, 
					int maxRays) {
  Matrix * Eqs;
  Polyhedron * Peqs;
  Matrix * M = Polyhedron2Constraints((*P));
  Matrix * Ct = Polyhedron2Constraints((*C));

  /* if the Minkowski representation is not computed yet, do not compute it in
     Constraints2Polyhedron */
  if (F_ISSET((*P), POL_VALID | POL_INEQUALITIES) && 
      (F_ISSET((*C), POL_VALID | POL_INEQUALITIES))) {
    FL_INIT(maxRays, POL_NO_DUAL);
  }
    
  Eqs = Constraints_Remove_parm_eqs(&M, &Ct, renderSpace, elimParms);
  Peqs = Constraints2Polyhedron(Eqs, maxRays);
  Matrix_Free(Eqs);

  /* particular case: no equality involving only parms is found */
  if (Eqs->NbRows==0) {
    Matrix_Free(M);
    Matrix_Free(Ct);
    return Peqs;
  }
  Polyhedron_Free(*P);
  Polyhedron_Free(*C);
  (*P) = Constraints2Polyhedron(M, maxRays);
  (*C) = Constraints2Polyhedron(Ct, maxRays);
  Matrix_Free(M);
  Matrix_Free(Ct);
  return Peqs;
} /* Polyhedron_Remove_parm_eqs */


/**
 * Given a matrix with m parameterized equations, compress the nb_parms
 * parameters and n-m variables so that m variables are integer, and transform
 * the variable space into a n-m space by eliminating the m variables (using
 * the equalities) the variables to be eliminated are chosen automatically by
 * the function.
 * @param M the constraints 
 * @param the number of parameters
 * @param validityLattice the the integer lattice underlying the integer
 * solutions.
*/
Matrix * full_dimensionize(Matrix const * M, int nb_parms, 
			   Matrix ** validityLattice) {
  Matrix * Eqs, * Ineqs;
  Matrix * Permuted_Eqs, * Permuted_Ineqs;
  Matrix * Full_Dim;
  Matrix * WVL; /* The Whole Validity Lattice (vars+parms) */
  unsigned int i,j;
  int nb_elim_vars;
  unsigned int * permutation, * permutation_inv;
  /* 0- Split the equalities and inequalities from each other */
  split_constraints(M, &Eqs, &Ineqs);

  /* 1- if the polyhedron is already full-dimensional, return it */
  if (Eqs->NbRows==0) {
    Matrix_Free(Eqs);
    (*validityLattice) = Identity_Matrix(nb_parms+1);
    return Ineqs;
  }
  nb_elim_vars = Eqs->NbRows;

  /* 2- put the vars to be eliminated at the first positions, 
     and compress the other vars/parms
     -> [ variables to eliminate / parameters / variables to keep ] */
  permutation = find_a_permutation(Eqs, nb_parms);
  if (dbgCompParm) {
    printf("Permuting the vars/parms this way: [ ");
    for (i=0; i< Eqs->NbColumns; i++) {
      printf("%d ", permutation[i]);
    }
    printf("]\n");
  }
  Permuted_Eqs = mpolyhedron_permute(Eqs, permutation);
  WVL = compress_parms(Permuted_Eqs, Eqs->NbColumns-2-Eqs->NbRows);
  if (dbgCompParm) {
    printf("Whole validity lattice: ");
    show_matrix(WVL);
  }
  mpolyhedron_compress_last_vars(Permuted_Eqs, WVL);
  Permuted_Ineqs = mpolyhedron_permute(Ineqs, permutation);
  if (dbgCompParm) {
    show_matrix(Permuted_Eqs);
  }
  Matrix_Free(Eqs);
  Matrix_Free(Ineqs);
  mpolyhedron_compress_last_vars(Permuted_Ineqs, WVL);
  if (dbgCompParm) {
    printf("After compression: ");
    show_matrix(Permuted_Ineqs);
  }
  /* 3- eliminate the first variables */
  if (!mpolyhedron_eliminate_first_variables(Permuted_Eqs, Permuted_Ineqs)) {
    fprintf(stderr,"full-dimensionize > variable elimination failed. \n"); 
    return NULL;
  }
  // show_matrix(Permuted_Eqs);
  if (dbgCompParm) {
    printf("After elimination of the variables: ");
    show_matrix(Permuted_Ineqs);
  }

  /* 4- get rid of the first (zero) columns, 
     which are now useless, and put the parameters back at the end */
  Full_Dim = Matrix_Alloc(Permuted_Ineqs->NbRows,
			  Permuted_Ineqs->NbColumns-nb_elim_vars);
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
  *validityLattice = Matrix_Alloc(nb_parms+1, nb_parms+1);
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++)
      value_assign((*validityLattice)->p[i][j], 
		   WVL->p[i][j]);
    value_assign((*validityLattice)->p[i][nb_parms], 
		 WVL->p[i][WVL->NbColumns-1]);
  }
  for (j=0; j< nb_parms; j++) 
    value_set_si((*validityLattice)->p[nb_parms][j], 0);
  value_assign((*validityLattice)->p[nb_parms][nb_parms], 
	       WVL->p[WVL->NbColumns-1][WVL->NbColumns-1]);

  /* 6- Clean up */
  Matrix_Free(WVL);
  return Full_Dim;
} /* full_dimensionize */

