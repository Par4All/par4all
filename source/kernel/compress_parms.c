// (C) B. Meister 12/2003
// LSIIT -ICPS 
// UMR 7005 CNRS
// Louis Pasteur University (ULP), Strasbourg, France

#include <polylib/polylib.h>
#include <polylib/compress_parms.h>

// utility function, scope = this file
static int gcd(int num, int den) {

  // take the absolute values of num and den
  int a = (num>0)?num:-num;
  int b = (den>0)?den:-den;

  while ((a>0)&&(b>0)) {
    if (a>b) a-=b;
    else if(b>a) b-=a;
    else return a;
  }
  if (a>0) return a;
  if (b>0) return b;
  if (a==0 || b==0) return 0;
  else {
    fprintf(stderr, "gcd(%d,%d) > control flow inconsistency",num, den);
    exit(1);
  }
}

int lcm(int num, int den) {

  // take the absolute values of num and den
  int a = (num>0)?num:-num;
  int b = (den>0)?den:-den;
  if (a==0) 
    if (b==0) return 1;
    else return b;
  if (b==0) return a;
  return (a*b/gcd(a,b));
}

// custom modulo, following the definition:
// a mod b = c equivalent_to: there exists a bigest nonnegative n in Z such that a = n|b| + c
// (the % C operator is not trusted here)
void b_modulo(Value g, Value a, Value b) {
  if (value_zero_p(b)) {value_assign(g,a); return;}
  if (value_neg_p(b)) value_oppose(b, b);
  if (value_posz_p(a)) {
    value_division(g,a,b);
    value_multiply(g,g,b);
    value_substract(g, a, g);
    return;
  }
  else {
    value_division(g, a, b);
    value_decrement(g, g);
    value_multiply(g, g, b);
    value_substract(g, a, g);
    //  g =a-(a/b-1)*b;
    if (value_eq(g,b)) value_set_si(g,0);
  }
}


//computes the modulo inverse of a modulo n
Value modulo_inverse(const Value a, const Value n) {
#ifdef GNUMP
  Value res;
  value_init(res);
  mpz_invert(res, a, n);
  return res;
#else
  if (gcd(a,n)!=1) return 0;
  // solve ax + ny = 1
  // the result is x
  Matrix * Pb = Matrix_Alloc(2,1);
  Pb->p[0][0] = a;
  Pb->p[1][0] = n;

  Matrix * H, * U, * Q;
  // computes U.Pb = H. (H = 1)
  right_hermite(Pb, &H, &U, &Q);
  Matrix_Free(Pb);
  Matrix_Free(H);
  Matrix_Free(Q);
  return U->p[0][0];
#endif
}

// given a full-row-rank nxm matrix M(made of m row-vectors), 
// computes the basis K (made of n-m column-vectors) of the integer kernel of the rows of M
// so we have: M.K = 0
// assumed: M is full row rank
Matrix * int_ker(Matrix * M) {
  Matrix *U, *Q, *H, *K;
  int i, j;

  // there is a non-null kernel if and only if the dimension m of the space spanned by the rows 
  // is inferior to the number n of variables 
  if (M->NbColumns <= M->NbRows) {
    K = Matrix_Alloc(M->NbColumns, 0);
    return K;
  }
  // computes MU = [H 0]
  left_hermite(M, &H, &Q, &U);
  
  // obviously, the Integer Kernel is made of the last n-m columns of U
  K=Matrix_Alloc(U->NbRows, U->NbColumns-M->NbRows);
  for (i=0; i< U->NbRows; i++)
    for(j=0; j< U->NbColumns-M->NbRows; j++) 
      value_assign(K->p[i][j], U->p[i][M->NbRows+j]);

  // clean up
  Matrix_Free(H);
  Matrix_Free(U);
  Matrix_Free(Q);
  return K;
}

// Compute the overall period of the variables I for (MI) mod |d|,
// where M is a matrix and |d| a vector
// Produce a diagonal matrix S = (s_k) where s_k is the overall period of i_k
Matrix * affine_periods(Matrix * M, Matrix * d) {
  Matrix * S;
  unsigned int i,j;
  Value tmp;
  value_init(tmp);
  // 1- compute the overall periods
  Value * periods = (Value *)malloc(sizeof(Value) * M->NbColumns);
  for(i=0; i< M->NbColumns; i++) 
    value_set_si(periods[i], 1);
  for (i=0; i<M->NbRows; i++) {
    for (j=0; j< M->NbColumns; j++) {
      Gcd(d->p[i][0], M->p[i][j], &tmp);
      value_division(tmp, d->p[i][0], tmp);
      B_Lcm(periods[j], tmp, &(periods[j]));
     }
  }
  value_clear(tmp);

  // 2- build S
  S = Matrix_Alloc(M->NbColumns, M->NbColumns);
  for (i=0; i< M->NbColumns; i++) 
    for (j=0; j< M->NbColumns; j++)
      if (i==j) value_assign(S->p[i][j],periods[j]);
      else value_set_si(S->p[i][j], 0);

  // 3- clean up
  free(periods);
  return S;
}

// given a matrix B' with m rows and m-vectors C' and d, computes the 
// basis of the integer solutions to (B'N+C') mod d = 0 (1).
// the Ns verifying the system B'N+C' = 0 are solutions of (1)
// K is a basis of the integer kernel of B: its column-vectors link two solutions of (1)
// Moreover, B'_iN mod d is periodic of period (s_ik):
// B'N mod d is periodic of period (s_k) = lcm_i(s_ik)
// The linear part of G is given by the HNF of (K | S), where S is the full-dimensional diagonal matrix (s_k)
// the constant part of G is a particular solution of (1)
// if no integer constant part is found, there is no solution and this function returns NULL.
Matrix * int_mod_basis(Matrix * Bp, Matrix * Cp, Matrix * d) {
  int nb_eqs = Bp->NbRows;
  unsigned int nb_parms=Bp->NbColumns;
  unsigned int i, j;

  //   a/ compute K and S
  Matrix * K = int_ker(Bp);
  Matrix * S = affine_periods(Bp, d);

  // show_matrix(K);
  // show_matrix(S);
  
  //   b/ compute the linear part of G : HNF(K|S)

  // fill K|S
  Matrix * KS = Matrix_Alloc(nb_parms, K->NbColumns+ nb_parms);
  for(i=0; i< KS->NbRows; i++) {
    for(j=0; j< K->NbColumns; j++) value_assign(KS->p[i][j], K->p[i][j]);
    for(j=0; j< S->NbColumns; j++) value_assign(KS->p[i][j+K->NbColumns], S->p[i][j]);
  }

  // show_matrix(KS);

  // HNF(K|S)
  Matrix * H, * U, * Q;
  left_hermite(KS, &H, &U, &Q);
  Matrix_Free(KS);
  Matrix_Free(U);
  Matrix_Free(Q);
  
  // printf("HNF(K|S) = ");show_matrix(H);

  // put HNF(K|S) in the p x p matrix S (which has already the appropriate size so we economize a Matrix_Alloc)
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++) 
      value_assign(S->p[i][j], H->p[i][j]);


  }
  Matrix_Free(H);

  //   c/ compute U_M.N'_0 = N_0: 
  Matrix * M = Matrix_Alloc(nb_eqs, nb_parms+nb_eqs);
  // N'_0 = M_H^{-1}.(-C' mod d), which must be integer (à verifier encore un peu mais je crois que c'est ok)
  // and where H_M = HNF(M) with M = (B' D) : M.U_M = [H_M 0]

  //       copy the B' part
  for (i=0; i< nb_eqs; i++) {
    for (j=0; j< nb_parms; j++) {
      value_assign(M->p[i][j], Bp->p[i][j]);
    }
  //       copy the D part
    for (j=0; j< nb_eqs; j++) {
      if (i==j) value_assign(M->p[i][j+nb_parms], d->p[i][0]);
      else value_set_si(M->p[i][j+nb_parms], 0);
    }
  }
  
  //       compute inv_H_M, the inverse of the HNF H of M = (B' D)
  left_hermite(M, &H, &Q, &U);
  Matrix * inv_H_M=Matrix_Alloc(nb_eqs, nb_eqs+1);
  //again, do a square Matrix from H, using the non-used Matrix Ha
  Matrix * Ha = Matrix_Alloc(nb_eqs, nb_eqs);
  for(i=0; i< nb_eqs; i++) {
    for(j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j], H->p[i][j]); 
    }
  }
  MatInverse(Ha, inv_H_M);
  Matrix_Free(Ha);
  Matrix_Free(H);
  Matrix_Free(Q); // only inv_H_M and U_M (alias U) are needed

  //       compute (-C') mod d
  Matrix * Cp_mod_d = Matrix_Alloc(nb_eqs, 1);
  for (i=0; i< nb_eqs; i++) {
    value_oppose(Cp->p[i][0], Cp->p[i][0]);
    b_modulo(Cp_mod_d->p[i][0], Cp->p[i][0], d->p[i][0]);
    //    Cp_mod_d->p[i][0] = b_modulo(-Cp->p[i][0], d->p[i][0]);
  }

  // Compute N'_0 = inv_H_M.((-C') mod d)
  //  actually compute (N' \\ 0) sot hat N = U^{-1}.(N' \\ 0)
  Matrix * Np_0= Matrix_Alloc(U->NbColumns, 1);
  for(i=0; i< nb_eqs; i++) 
    {
      value_set_si(Np_0->p[i][0], 0);
      for(j=0; j< nb_eqs; j++) {
	value_addmul(Np_0->p[i][0], inv_H_M->p[i][j], Cp_mod_d->p[j][0]);
      }
    }
  for(i=nb_eqs; i< U->NbColumns; i++) value_set_si(Np_0->p[i][0], 0);
  

  // it is still needed to divide the rows of N'_0 by the common denominator of the rows of H_M
  // if these rows are not divisible, there is no integer N'_0 so return NULL
  for (i=0; i< nb_eqs; i++) {

#ifdef GNUMP
    if (mpz_divisible_p(Np_0->p[i][0], inv_H_M->p[i][nb_eqs])) mpz_divexact(Np_0->p[i][0], Np_0->p[i][0], inv_H_M->p[i][nb_eqs]);
#else
    if (!(Np_0->p[i][0]%inv_H_M->p[i][nb_eqs])) Np_0->p[i][0]/=inv_H_M->p[i][nb_eqs];
#endif
    else {
      Matrix_Free(S);
      Matrix_Free(inv_H_M);
      Matrix_Free(Np_0);
      fprintf(stderr, "int_mod_basis > No particular solution: polyhedron without integer points.\n");
      return NULL;
    }
  }
  // show_matrix(Np_0);

  // now compute the actual particular solution N_0 = U_M. N'_0
  Matrix * N_0 = Matrix_Alloc(U->NbColumns, 1);
  // OPT: seules les nb_eq premières valeurs de N_0 sont utiles en fait.
  Matrix_Product(U, Np_0, N_0);
  // show_matrix(N_0);
  Matrix_Free(Np_0);
  Matrix_Free(U);

  // build the whole compression matrix: 
  Matrix * G = Matrix_Alloc(S->NbRows+1, S->NbRows+1);
  for (i=0; i< S->NbRows; i++) {
    for(j=0; j< S->NbRows; j++) 
      value_assign(G->p[i][j], S->p[i][j]);
    value_assign(G->p[i][S->NbRows], N_0->p[i][0]);
  }

  for (j=0; j< S->NbRows; j++) value_set_si(G->p[S->NbRows][j],0);
  value_set_si(G->p[S->NbRows][S->NbRows],1);

  // clean up
  Matrix_Free(S);
  Matrix_Free(N_0);
  return G;
}
// end of int_mod_basis()


// utiliy function: given a matrix of constraints AI+BN+C, extract the part of A, corresponding to the variables.
Matrix * get_linear_part(Matrix const * E, int nb_parms){ 
  unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * A = Matrix_Alloc(nb_eqs, nb_vars);
  k=0;
  for (i=0; i< E->NbRows; i++) {
    for (j=0; j< nb_vars; j++)
      value_assign(A->p[i][j],E->p[i][j+1]);
  }
  return A;
}

// utiliy function: given a matrix of constraints AI = -BN-C, extract the part of -B, corresponding to the parametrs.
Matrix * get_parameter_part(Matrix const * E, int nb_parms) {
  unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * B = Matrix_Alloc(nb_eqs,nb_parms);
  k=0;
  // well a priori the minus is unuseful here, but
  for(i=0; i< E->NbRows; i++) {
    for(j=0; j< nb_parms; j++)
      value_oppose(B->p[i][j], E->p[i][1+nb_vars+j]);
  }
  return B;
}

// utiliy function: given a matrix of constraints (0 or 1)AI= -BN-C, extract the part of -C corresponding to the equalities.
Matrix * get_constant_part(Matrix const * E, int nb_parms) {
  unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * C = Matrix_Alloc(nb_eqs,1);
  k=0;
  for(i=0; i< E->NbRows; i++) {
    value_oppose(C->p[i][0], E->p[i][E->NbColumns-1]);
  }
  return C;
}

// utility function: given a matrix containing the equations AI+BN+C=0, compute the HNF of A : A = [Ha 0].Q and return :  
// . B'= H^-1.(-B) 
// . C'= H^-1.(-C)
// . U = Q^-1 (-> return value)
// . D, where Ha^-1 = D^-1.H^-1 with H and D integer matrices 
// in fact, as D is diagonal, we return d, a column-vector
Matrix * extract_funny_stuff(Matrix const * E, int nb_parms, Matrix ** Bp, Matrix **Cp, Matrix **d) {
unsigned int i,j, k, nb_eqs=E->NbRows;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix * A, * Ap, * Ha, * U, * Q, * H, *B, *C;

  // particular case: 
  if (nb_eqs==0) {
    *Bp = Matrix_Alloc(0, E->NbColumns);
    *Cp = Matrix_Alloc(0, E->NbColumns);
    *d = NULL;
    return NULL;
  }

  // 1- build A
  A = get_linear_part(E, nb_parms);
  // show_matrix(A);

  // 2- Compute Ha^-1, where Ha is the left HNF of A
  //   a/ Compute H = [Ha 0]
  left_hermite(A, &H, &Q, &U);
  Matrix_Free(A);
  Matrix_Free(Q);
  
  //   b/ just keep the m x m matrix Ha
  Ha = Matrix_Alloc(nb_eqs, nb_eqs);
  for (i=0; i< nb_eqs; i++) {
    for (j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j],H->p[i][j]);
    }
  }
  Matrix_Free(H);

  // show_matrix(Ha);

  //  c/ Invert Ha
  Matrix * Ha_pre_inv = Matrix_Alloc(nb_eqs, nb_eqs+1);
  if(!MatInverse(Ha, Ha_pre_inv)) { 
    fprintf(stderr,"extract_funny_stuff > Matrix Ha is non-invertible.");
  }
  // store back Ha^-1  in Ha, to spare a MatrixAlloc/MatrixFree
  for(i=0; i< nb_eqs; i++) {
    for(j=0; j< nb_eqs; j++) {
      value_assign(Ha->p[i][j],Ha_pre_inv->p[i][j]);
    }
  }
  // show_matrix(Ha_pre_inv);
  // the diagonal elements of D are stored in the last column of Ha_pre_inv (property of MatInverse).
  (*d) = Matrix_Alloc(Ha_pre_inv->NbRows, 1);

  for (i=0; i< Ha_pre_inv->NbRows; i++) value_assign((*d)->p[i][0], Ha_pre_inv->p[i][Ha_pre_inv->NbColumns-1]);
 
  // show_matrix(Ha);
  // show_matrix(*d);
  Matrix_Free(Ha_pre_inv);

  // 3- Build B'and C'
  // compute B'
  B = get_parameter_part(E, nb_parms);
  (*Bp) = Matrix_Alloc(B->NbRows,B->NbColumns);
  // show_matrix(B);
  Matrix_Product(Ha, B, (*Bp));
  // show_matrix(*Bp);
  Matrix_Free(B);
  
  // compute C'
  C = get_constant_part(E, nb_parms);
  // show_matrix(C);
  (*Cp) = Matrix_Alloc(nb_eqs, 1);
  Matrix_Product(Ha, C, (*Cp));
  // show_matrix(*Cp);
  Matrix_Free(C);

  Matrix_Free(Ha);
  return U;
}
  

// given a parameterized constraints matrix with m equalities, computes the compression matrix 
// G such that there is an integer solution in the variables space for each value of N', with 
// N = G N' (N are the "nb_parms" parameters)
Matrix * compress_parms(Matrix * E, int nb_parms) {
  unsigned int i,j, k, nb_eqs=0;
  int nb_vars=E->NbColumns - nb_parms -2;
  Matrix *U, *d, *Bp, *Cp;
  U = extract_funny_stuff(E, nb_parms, &Bp, & Cp, &d);

  // particular case where there is no equation
  if (U==NULL) return Identity_Matrix(nb_parms+1);

  Matrix_Free(U);
  // The compression matrix N = G.N' must be such that (B'N+C') mod d = 0 (1)

  // the Ns verifying the system B'N+C' = 0 are solutions of (1)
  // K is a basis of the integer kernel of B: its column-vectors link two solutions of (1)
  // Moreover, B'_iN mod d is periodic of period (s_ik):
  // B'N mod d is periodic of period (s_k) = lcm_i(s_ik)
  // The linear part of G is given by the HNF of (K | S), where S is the full-dimensional diagonal matrix (s_k)
  // the constant part of G is a particular solution of (1)
  // if no integer constant part is found, there is no solution.

  Matrix * G = int_mod_basis(Bp, Cp, d);
  Matrix_Free(Bp);
  Matrix_Free(Cp);
  Matrix_Free(d);
  return G;
}

// given a matrix with m parameterized equations, compress the nb_parms parameters and n-m variables so that m variables are integer,
// and transform the variable space into a n-m space by eliminating the m variables (using the equalities)
// the variables to be eliminated are chosen automatically by the function 
Matrix * full_dimensionize(Matrix const * M, int nb_parms, Matrix ** Validity_Lattice) {
  Matrix * Eqs, * Ineqs;
  Matrix * Permuted_Eqs, * Permuted_Ineqs;
  Matrix * Full_Dim;
  Matrix * Whole_Validity_Lattice;
  unsigned int i,j;
  int nb_elim_vars;
  int * permutation, * permutation_inv;
  // 0- Split the equalities and inequalities from each other
  split_constraints(M, &Eqs, &Ineqs);

  // 1- if the polyhedron is already full-dimensional, return it
  if (Eqs->NbRows==0) {
    Matrix_Free(Eqs);
    (*Validity_Lattice) = Identity_Matrix(nb_parms+1);
    return Ineqs;
  }
  nb_elim_vars = Eqs->NbRows;

  // 2- put the vars to be eliminated at the first positions, and compress the other vars/parms
  // -> [ variables to eliminate / parameters / variables to keep ]
  permutation = find_a_permutation(Eqs, nb_parms);
  Permuted_Eqs = mpolyhedron_permute(Eqs, permutation);
  Whole_Validity_Lattice = compress_parms(Permuted_Eqs, Eqs->NbColumns-2-Eqs->NbRows);
  mpolyhedron_compress_last_vars(Permuted_Eqs, Whole_Validity_Lattice);
  Permuted_Ineqs = mpolyhedron_permute(Ineqs, permutation);
  Matrix_Free(Eqs);
  Matrix_Free(Ineqs);
  mpolyhedron_compress_last_vars(Permuted_Ineqs, Whole_Validity_Lattice);

  // 3- eliminate the first variables
  if (!mpolyhedron_eliminate_first_variables(Permuted_Eqs, Permuted_Ineqs)) {
    fprintf(stderr,"full-dimensionize > variable elimination failed. \n"); 
    return NULL;
  }

  // 4- get rid of the first (zero) columns, which are now useless, and put the parameters back at the end
  Full_Dim = Matrix_Alloc(Permuted_Ineqs->NbRows, Permuted_Ineqs->NbColumns-nb_elim_vars);
  for (i=0; i< Permuted_Ineqs->NbRows; i++) {
    value_set_si(Full_Dim->p[i][0], 1);
    for (j=0; j< nb_parms; j++) 
      value_assign(Full_Dim->p[i][j+Permuted_Ineqs->NbColumns-nb_parms-2-nb_elim_vars+1], Permuted_Ineqs->p[i][j+nb_elim_vars+1]);
    for (j=0; j< Permuted_Ineqs->NbColumns-nb_parms-2-nb_elim_vars; j++) 
      value_assign(Full_Dim->p[i][j+1], Permuted_Ineqs->p[i][nb_elim_vars+nb_parms+j+1]);
    value_assign(Full_Dim->p[i][Full_Dim->NbColumns-1], Permuted_Ineqs->p[i][Permuted_Ineqs->NbColumns-1]);
  }
  Matrix_Free(Permuted_Ineqs);
  // return Full_Dim; 

  // 4- Un-permute (so that the parameters are at the end as usual)
  // -> [variables (mixed) / parameters ]
  /*permutation_inv = permutation_inverse(permutation, Permuted_Ineqs->NbColumns-1);
  free(permutation);
  Ineqs = mpolyhedron_permute(Permuted_Ineqs, permutation_inv);
  mpolyhedron_simplify(Ineqs);
  free(permutation_inv); */
  
  // 5- Keep only the the validity lattice restricted to the parameters
  *Validity_Lattice = Matrix_Alloc(nb_parms+1, nb_parms+1);
  for (i=0; i< nb_parms; i++) {
    for (j=0; j< nb_parms; j++)
      value_assign((*Validity_Lattice)->p[i][j], Whole_Validity_Lattice->p[i][j]);
    value_assign((*Validity_Lattice)->p[i][nb_parms], Whole_Validity_Lattice->p[i][Whole_Validity_Lattice->NbColumns-1]);
  }
  for (j=0; j< nb_parms; j++) value_set_si((*Validity_Lattice)->p[nb_parms][j], 0);
  value_assign((*Validity_Lattice)->p[nb_parms][nb_parms], Whole_Validity_Lattice->p[Whole_Validity_Lattice->NbColumns-1][Whole_Validity_Lattice->NbColumns-1]);

  // 6- Clean up 
  Matrix_Free(Whole_Validity_Lattice);
  return Full_Dim;
  //  return Ineqs;
} // full_dimensionize



// computes the compression matrices G and G2 so that: 
// - with G, I is integer for any integer N
// - with G2, I2 is integer for any integer N, I1
// input : - M, the equations 
//         - the number of parameters
// output: - the global compression 
//         - G (compression for the parameters)
Matrix * compress_for_full_dim(Matrix * M, int nb_parms, Matrix **  compression, Matrix ** G) {
  Matrix * U, * Bp, * Cp, * d;
  int const nb_vars=M->NbColumns-nb_parms-2;
  int i,j,k;

  // compute the compression of the parameters space, so that there is one integer solution in the variable space for any value of the parms
  // (ie. do what compress_parms does)
  (*G) = compress_parms(M, nb_parms);

  show_matrix(*G);
  // 1- actually do the compression, while changing the order of the variables and the parameters
  Matrix * Mp = Matrix_Alloc(M->NbRows, M->NbColumns);

  // a- put the variables to be eliminated at the first place
  for (i=0; i< Mp->NbRows; i++) {
    value_set_si(Mp->p[i][0], 0);
    for (j=0; j< Mp->NbRows; j++) {
      value_assign(Mp->p[i][j+1], M->p[i][j+nb_vars-Mp->NbRows+1]);
    }
  }

  // b- apply G to the parameters and constant space and place the parameters as second indices
  for (i=0; i< M->NbRows;i++) {
    for (j=0; j< nb_parms+1; j++) {
      value_multiply(Mp->p[i][j+Mp->NbRows+1], M->p[i][nb_vars+1], (*G)->p[0][j]);
      for (k=1; k< nb_parms+1; k++) {
	value_addmul(Mp->p[i][j+Mp->NbRows+1], M->p[i][k+nb_vars+1], (*G)->p[k][j]);
      }
    }
    value_multiply(Mp->p[i][Mp->NbColumns-1], M->p[i][nb_vars+1], (*G)->p[0][(*G)->NbColumns-1]);
    for (k=1; k<nb_parms+1; k++) {
      value_addmul(Mp->p[i][Mp->NbColumns-1], M->p[i][k+nb_vars+1], (*G)->p[k][(*G)->NbColumns-1]);
    }
  }

  // c- put the variables to stay at the end (but before the constants)
  for (i=0; i< Mp->NbRows; i++) {
    for (j=0; j< nb_vars-Mp->NbRows; j++) {
      value_assign(Mp->p[i][j+nb_parms+Mp->NbRows+1], M->p[i][j+1]);
    }
  }
     
  show_matrix(Mp);

  // simplify M'
  Value cur_gcd;
  value_init(cur_gcd);
  for (i=0; i< Mp->NbRows; i++) {
    value_assign(cur_gcd, Mp->p[i][1]);
    for (j=1; j< Mp->NbColumns; j++)
      Gcd(cur_gcd, Mp->p[i][j], &cur_gcd);
    for (j=1; j< Mp->NbColumns; j++) {
#ifdef GNUMP
      mpz_divexact(Mp->p[i][j], Mp->p[i][j], cur_gcd);
#else
      Mp->p[i][j]/=cur_gcd;
#endif
    }
  }
  value_clear(cur_gcd);
  printf("M' simplified : ");show_matrix(Mp);

  // 2- do the same computation on M', with I1 and N as parameters
  // the corresponding nb_parms is then nb_parms + nb_vars-Mp->NbRows
  Matrix * G2=compress_parms(Mp, nb_parms+nb_vars-Mp->NbRows);
  show_matrix(G2);
  
  // 3- Combine G, the permutation (I1, N) -> (N, I1) and G2 to give the resulting compression
  // the combination is: G.Pi.G2.Pi^{-1}
  // Pi.G2 is a permutation of the rows of G2
  // So we can apply a permutation of the rows of G2 to G to get the resulting compression
  // also note that G applies on parameters
  (*compression) = Matrix_Alloc(nb_parms+nb_vars-Mp->NbRows+1, nb_parms+nb_vars-Mp->NbRows+1);
  // a- permute the rows and columns of G2 :
  //  1/ I1(I1)
  for(i=0; i< nb_vars-Mp->NbRows;  i++) 
    for (j=0; j< nb_vars-Mp->NbRows;  j++) 
      value_assign((*compression)->p[i][j], G2->p[i+nb_parms][j+nb_parms]);

  // 2/ I1(N)
  for (i=0; i< nb_vars-Mp->NbRows; i++) 
  for (j=0; j< nb_parms; j++) 
    value_assign((*compression)->p[i][j+nb_vars-Mp->NbRows], G2->p[i+nb_parms][j]);

  // 3/ N(N)
  for (i=0; i< nb_parms; i++) 
    for (j=0; j< nb_parms; j++) 
      value_assign((*compression)->p[i+nb_vars-Mp->NbRows][j+nb_vars-Mp->NbRows], G2->p[i][j]);

  // 4/ the part N(I1) is zero, as G2 is inferior triangular
  for(i=0; i< nb_parms; i++) 
    for (j=0; j< nb_vars-Mp->NbRows; j++) 
      value_assign((*compression)->p[i+nb_vars-Mp->NbRows][j], G2->p[i][j+nb_parms]);

  // the constant part is affected by the row permutation
  for (i=0; i< nb_parms; i++)
    value_assign((*compression)->p[i+nb_vars-Mp->NbRows][(*compression)->NbColumns-1], G2->p[i][nb_parms+nb_vars-Mp->NbRows]);
  for (i=0; i< nb_vars-Mp->NbRows; i++) 
    value_assign((*compression)->p[i][nb_parms+nb_vars-Mp->NbRows], G2->p[i+nb_parms][nb_parms+nb_vars-Mp->NbRows]);
  for (i=0; i< nb_vars-Mp->NbRows+nb_parms; i++) value_set_si((*compression)->p[(*compression)->NbRows-1][i], 0);
   value_set_si((*compression) ->p[nb_parms+nb_vars-Mp->NbRows][nb_parms+nb_vars-Mp->NbRows], 1);

  // b- compute G . permuted G2
  // OPT: il faudrait combiner les 2 opérations (permutation + application de G) en 1 seule
  Matrix * G_tmp = Matrix_Alloc(nb_parms+1, nb_parms+1);
  for (i=0; i< nb_parms+1; i++) {
    for (j=0; j< nb_parms+1; j++) { 
      value_set_si(G_tmp->p[i][j], 0);
      for (k=0; k< nb_parms+1; k++)
	value_addmul(G_tmp->p[i][j], (*G)->p[i][k],(*compression)->p[nb_vars-Mp->NbRows+k][nb_vars-Mp->NbRows+j]);
    }
  }

  for (i=0; i< nb_parms+1; i++) 
    for (j=0; j< nb_parms+1; j++) 
      value_assign((*compression)->p[i+nb_vars-Mp->NbRows][j+nb_vars-Mp->NbRows], G_tmp->p[i][j]);

  Matrix_Free(G2);
  Matrix_Free(G_tmp);

  return Mp;
} // compress_for_full_dim


// computes a one-pass compression so that I2 is integer for any integer value of I1 and N
// vars_to_eliminate is a list of e variables to keep, where e is the number of equations (M->NbRows)
// returns the full-dimensional polyhedron
Matrix * compress_to_full_dim2(Matrix * Eqs, Matrix * Ineqs, int nb_parms, unsigned int * vars_to_keep, Matrix ** compression, Matrix ** Cmp_Eqs) {
  int i,j,k;
  assert (Eqs&&Ineqs);
  unsigned int nb_keep = Eqs->NbColumns-2-nb_parms - Eqs->NbRows;

  // 1- Permute the variables so that the vars to be eliminated are at the first positions
  //   a- Compute the permutation matrix, which shifts the variables to be eliminated at the first ranks and swaps parms with the vars to be kept.
  unsigned int * permutation = permutation_for_full_dim2(vars_to_keep, nb_keep, Eqs->NbColumns-2, nb_parms);

   // checkpoint
  printf("permutation = [ ");
    for (i=0; i< Eqs->NbColumns-1; i++) printf("%ud ",permutation[i]);
  printf("]");

  //   b- do the permutation
  Matrix * Perm_Eqs = mpolyhedron_permute(Eqs, permutation);
  show_matrix(Perm_Eqs);


  // 2- Compress (N, I_1) so that I_2 is integer
  //   a- using the equalities, compute the compression
  (*compression) = compress_parms(Perm_Eqs, nb_parms+nb_keep);
  show_matrix((*compression));

  //   b- compress the equalities
  Matrix * tmp = Matrix_Alloc(1, nb_parms+nb_keep+1);
  for (i=0; i< Perm_Eqs->NbRows; i++) {
    for (j=0; j< nb_parms+nb_keep+1; j++) {
      value_multiply(tmp->p[0][j], Perm_Eqs->p[i][Perm_Eqs->NbRows+1], (*compression)->p[0][j]);
      for (k=1; k< nb_parms+nb_keep+1; k++) {
	value_addmul(tmp->p[0][j], Perm_Eqs->p[i][k+Eqs->NbRows+1],(*compression)->p[k][j]);
      }
    }
    for (j=0; j< nb_parms+nb_keep+1; j++) value_assign(Perm_Eqs->p[i][j+Eqs->NbRows+1], tmp->p[0][j]);
  }
  printf("Compressed Permuted Eqs: "); show_matrix(Perm_Eqs);
  
  //   c- compress the inequalities
  Matrix * Perm_Ineqs = mpolyhedron_permute(Ineqs, permutation);
  for (i=0; i< Perm_Ineqs->NbRows; i++) {
    for (j=0; j< nb_parms+nb_keep+1; j++) {
      value_multiply(tmp->p[0][j], Perm_Ineqs->p[i][Perm_Ineqs->NbRows+1], (*compression)->p[0][j]);
      for (k=1; k< nb_parms+nb_keep+1; k++) {
	value_addmul(tmp->p[0][j], Perm_Ineqs->p[i][k+Ineqs->NbRows+1],(*compression)->p[k][j]);
      }
    }
    for (j=0; j< nb_parms+nb_keep+1; j++) value_assign(Perm_Ineqs->p[i][j+Ineqs->NbRows+1], tmp->p[0][j]);
  }
  printf("Compressed Permuted Eqs: "); show_matrix(Perm_Ineqs);
  
  // 3- Using the equalities, row-eliminate the (first, as they are positionned on purpose) variables
  for(i=0; i< Perm_Eqs->NbRows; i++) {
    for (j=0; (j< Perm_Eqs->NbRows)&&(value_zero_p(Perm_Eqs->p[j][i+1])) ; j++);
    if (j==Perm_Eqs->NbRows) { fprintf(stderr, "Argh: non full-row-rank equalities matrix");1/0; }
    for (k=0; k<Perm_Eqs->NbRows; k++) {
      if (k!=j) eliminate_var_with_constr(Perm_Eqs, j, Perm_Eqs, k, i);
    }
    for (k=0; k< Perm_Ineqs->NbRows; k++) eliminate_var_with_constr(Perm_Eqs, j, Perm_Ineqs, k, i);
  }


  // 4- Unpermute the variables
  unsigned int * inv_perm =permutation_inverse(permutation, Eqs->NbColumns-1);
  printf("inv_perm = [ ");
    for (i=0; i< Eqs->NbColumns-1; i++) printf("%ud ",inv_perm[i]);
  printf("]");
  Matrix * Unperm_Perm_Eqs = mpolyhedron_permute(Perm_Eqs, inv_perm);
  show_matrix(Unperm_Perm_Eqs);
  Matrix * Unperm_Perm_Ineqs = mpolyhedron_permute(Perm_Ineqs, inv_perm);
  show_matrix(Unperm_Perm_Ineqs);


  // 5- Compute the full-dimensional polyhedron by keeping only (I1,N)
  Matrix * Full_Dim = Matrix_Alloc(Unperm_Perm_Ineqs->NbRows, Unperm_Perm_Ineqs->NbColumns-Eqs->NbRows);
  for (i=0; i< Unperm_Perm_Ineqs->NbRows; i++) {
    value_set_si(Full_Dim->p[i][0], 1);
    for (j=0; j< Unperm_Perm_Ineqs->NbColumns-Eqs->NbRows; j++) {
      value_assign(Full_Dim->p[i][j+1], Unperm_Perm_Ineqs->p[i][vars_to_keep[j]+1]);
    }
    for (j=0; j< nb_parms+1; j++) {
      value_assign(Full_Dim->p[i][j+Unperm_Perm_Ineqs->NbColumns-nb_parms-Eqs->NbRows-1], Unperm_Perm_Ineqs->p[i][j+Unperm_Perm_Ineqs->NbColumns-nb_parms-1]);
    }
  }
  mpolyhedron_simplify(Full_Dim);
  show_matrix(Full_Dim);
  return Full_Dim;
}


// in: Eqs = the original equations
//     Full_Dim = the full-dimensional veriosn of the inequalities (with n-m variables)
//     nb_parms: the number of parameters
//     vars_to_keep: the variables kept while projecting the polyhedron
//     compression: the compression made for projecting correctly the polyhedron.
Matrix * expand_to_non_full_dim2(Matrix * Eqs, Matrix * Full_Dim, int nb_parms, unsigned int * vars_to_keep, Matrix * compression) {
  int nb_keep = Full_Dim->NbColumns-nb_parms-2-Eqs->NbRows;
  int i,j,k;
  // 1- Transform Ineqs so that we can unapply the compression
  //   a- compute the permutation 
  Matrix * Non_Full_Dim = Matrix_Alloc(Full_Dim->NbRows, Full_Dim->NbColumns+Eqs->NbRows);
  unsigned int * permutation = permutation_for_full_dim2(vars_to_keep, nb_keep, Eqs->NbColumns-2, nb_parms);

  //   b- put the formerly removed variables at their original place
  for (i=0; i< Full_Dim->NbRows; i++) {
    Non_Full_Dim->p[i][0] = Full_Dim->p[i][0];
    for (j=0; j< nb_keep+Eqs->NbRows; i++) value_set_si(Non_Full_Dim->p[i][j+1], 0);
    for (j=0; j< nb_keep; j++) value_assign(Non_Full_Dim->p[i][vars_to_keep[j]+1], Full_Dim->p[i][j+1]);
    for (j=0; j< nb_parms+1; j++) value_assign(Non_Full_Dim->p[i][Eqs->NbColumns-nb_parms-1+j], Full_Dim->p[i][nb_keep+1+j]);
  }
  //   c- permute -> the formerly removed variables are at the beginning
  Matrix * Perm_NFD = mpolyhedron_permute(Non_Full_Dim, permutation);

  // 2- right-apply compression^{-1} (expansion) to Perm_NFD
  //   a- compute the expansion
  Matrix * tmp = Matrix_Copy(compression);
  Matrix * expansion = Matrix_Alloc(compression->NbRows, compression->NbColumns+1);
  Matrix_Free(tmp);
  MatInverse(tmp, expansion);
  //   b- as it is rational, put it to the same denominator
  Value cur_gcd; value_init(cur_gcd);
  Value factor; value_init(factor);
  for (i=0; i< expansion->NbRows; i++) Gcd(cur_gcd, expansion->p[i][expansion->NbColumns-1], & cur_gcd);
  for (i=0; i< expansion->NbRows; i++) {
    value_division(factor, cur_gcd, expansion->p[i][expansion->NbColumns-1]);
    for (j=0; j< expansion->NbColumns; j++) value_multiply(expansion->p[i][j], expansion->p[i][j], factor);
  }
  value_clear(cur_gcd);
  //   c- apply the expansion to the vars and parms in Perm_NFD
  for (i=0; i< Perm_NFD->NbRows; i++) {
    for (j=0; j< nb_parms+nb_keep+1; j++) {
      value_multiply(tmp->p[0][j], Perm_NFD->p[i][Eqs->NbRows+1], expansion->p[0][j]);
      for (k=1; k< nb_parms+nb_keep+1; k++) {
	value_addmul(tmp->p[0][j], Perm_NFD->p[i][k+Eqs->NbRows+1], expansion->p[k][j]);
      }
    }
    for (j=0; j< nb_parms+nb_keep+1; j++) value_assign(Perm_NFD->p[i][j+Eqs->NbRows+1], tmp->p[0][j]);
  }
  printf("Expanded Permuted Eqs: "); show_matrix(Perm_NFD);

  //   d- unpemrute -> all the variables and parameters are at their original position
  unsigned int * inv_perm =permutation_inverse(permutation, Eqs->NbColumns-1);
  printf("inv_perm = [ ");
  for (i=0; i< Perm_NFD->NbColumns-1; i++) printf("%ud ",inv_perm[i]);
  printf("]");
  Matrix * Unperm_Perm_NFD = mpolyhedron_permute(Perm_NFD, inv_perm);
  free(inv_perm);
  Matrix_Free(Perm_NFD);
  printf("Unpermuted expanded permuted Ineqs: "); show_matrix(Unperm_Perm_NFD);
  return Unperm_Perm_NFD;
}



// uses a set of equalitites to eliminate the last variables from a system of inequalitiesx.
// NYI : GMP version (with Values)
Matrix * eliminate_last_variables(Matrix * Eqs, Matrix * Ineqs, int nb_vars) {
  int cur_variable, cur_constraint, cur_lcm, a, b, sb, mul_a, mul_b, i, k;
  // projected polyhedron
  Matrix * proj = Matrix_Alloc(Ineqs->NbRows, Ineqs->NbColumns-Eqs->NbRows);

  // for each variable to be eliminated (with the help of an equation)
  for (cur_variable = nb_vars-Eqs->NbRows; cur_variable <nb_vars; cur_variable++) {

    // 0- find the equality with the first non-zero coef
    for (cur_constraint = 0; cur_constraint<Eqs->NbRows && Eqs->p[cur_constraint][cur_variable]==0; cur_constraint++);
    assert(cur_constraint<Eqs->NbRows);
    a = Eqs->p[cur_constraint][cur_variable+1];

    // 1- eliminate the variable in Eqs
    for (i=0; i< Eqs->NbRows; i++) {
      if (i!=cur_constraint) {
	b = Eqs->p[i][cur_variable+1];
	if (b!=0) {
	  cur_lcm = lcm(a, b);
	  // multiplication factor for the current constraint
	  mul_a = abs(cur_lcm / b);
	  sb = abs(b) / b;
	  // multiplication factor for the constraint
	  mul_b = cur_lcm / a * sb;
	  for (k=1; k<Eqs->NbColumns; k++) {
	    Eqs->p[i][k] = Eqs->p[i][k] * mul_a - Eqs->p[cur_constraint][k] * mul_b;
	  }
	}
      }
    }
    // 2- eliminate the variable in Ineqs
    for (i=0; i< Ineqs->NbRows; i++) {
      b = Ineqs->p[i][cur_variable+1];
      if (b!=0) {
	cur_lcm = lcm(a, b);
	// multiplication factor for the current constraint
	mul_a = abs(cur_lcm / b);
	sb = abs(b) / b;
	// multiplication factor for the constraint
	mul_b = cur_lcm / a * sb;
	proj->p[i][0] = 1;
	for (k=1; k<nb_vars-Eqs->NbRows+1; k++) {
	  proj->p[i][k] = Ineqs->p[i][k] * mul_a - Eqs->p[cur_constraint][k] * mul_b;
	}
      	for (k=nb_vars+1; k< Ineqs->NbColumns; k++) {
	  proj->p[i][k-Eqs->NbRows] = Ineqs->p[i][k] * mul_a - Eqs->p[cur_constraint][k] * mul_b;
     	} 
      }
      // if the coef is zero, do not modify the constraint
      else {
	for(k=0; k< nb_vars-Eqs->NbRows+1; k++) 
	  proj->p[i][k] = Ineqs->p[i][k];
	for(k=nb_vars+1; k< Ineqs->NbColumns; k++) 
	  proj->p[i][k-Eqs->NbRows] = Ineqs->p[i][k];
      }
    }
  }
  return proj;
}


