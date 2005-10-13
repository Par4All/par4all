// Permutations on matrices
// Matrices are seen either as transformations (mtransformation) or as polyhedra (mpolyhedron)
// B. Meister
// C code

#include <polylib/matrix_permutations.h>

// utility function : bit count (i know, there are faster methods)
unsigned int nb_bits(unsigned long long int x) {
  unsigned int i,n=0;
  unsigned long long int y=x;
  for (i=0; i< 64; i++) {
    n+=y%2;
    y>>=1;
  }
  return n;
}


// gives the inverse permutation vector of a permutation vector
unsigned int * permutation_inverse(unsigned int * perm, unsigned int nb_elems) {
  int i;
  unsigned int * inv_perm = (unsigned int *)malloc(sizeof(unsigned int) * nb_elems);
  for (i=0; i< nb_elems; i++) inv_perm[perm[i]] = i;
  return inv_perm;
}
  

// Given a linear tranformation on initial variables, and a variable permutation, compute the tranformation for the permuted variables.
// perm is a vector giving the new "position of the k^th variable, k \in [1..n]
// we can call it a "permutation vector" if you wish
// transf[x][y] -> permuted[permutation(x)][permutation(y)]
Matrix * mtransformation_permute(Matrix * transf, unsigned int * permutation) {
  Matrix * permuted;
  unsigned int i,j;
  // the transformation is supposed to be from Q^n to Q^n, so a square matrix.
  assert(transf->NbRows==transf->NbColumns);
  permuted = Matrix_Alloc(transf->NbRows, transf->NbRows);
  for (i= 0; i< transf->NbRows; i++) {
    for (j= 0; j< transf->NbRows; j++) {
      value_assign(permuted->p[permutation[i]][permutation[j]], transf->p[i][j]);
    }
  }
  return permuted;
}

// permutes the variables of a matrix seen as a polyhedron
Matrix * mpolyhedron_permute(Matrix * polyh, unsigned int * permutation) {
  unsigned int i,j;
  Matrix * permuted = Matrix_Alloc(polyh->NbRows, polyh->NbColumns);
  for (i= 0; i< polyh->NbRows; i++) {
    value_assign(permuted->p[i][0], polyh->p[i][0]);
    for (j= 1; j< polyh->NbColumns; j++) {
      value_assign(permuted->p[i][permutation[j-1]+1], polyh->p[i][j]);
    }
  }
  return permuted;
}


// find a valid permutation : for a set of m equations, find m variables that will be put at the beginning (to be eliminated)
// it must be possible to eliminate these variables : the submatrix built with their columns must be full-rank.
// brute force method, that tests all the combinations until finding one which works.
// LIMITATIONS : up to x-1 variables, where the long long format is x-1 bits (often 64 in year 2005).
unsigned int * find_a_permutation(Matrix * Eqs, unsigned int nb_parms) {
  unsigned int i, j, k;
  int nb_vars = Eqs->NbColumns-nb_parms-2;
  unsigned long long int combination;
  unsigned int * permutation = (unsigned int *)malloc(sizeof(unsigned int) * Eqs->NbColumns-1);
  int found = 0;
  Matrix * Square_Mat = Matrix_Alloc(Eqs->NbRows, Eqs->NbRows);
  Matrix * M, * H, * Q, *U;

  // generate all the combinations of Eqs->NbRows variables (-> bits to 1 in the word "combination") among nb_vars
  // WARNING : we assume here that we have not more than 64 variables...
  // you may convert it to use GNU MP to set it to an infinite number of bits
  for (combination = ((unsigned long long int) 1<<(Eqs->NbRows))-1; (combination < ((unsigned long long int) 1 << nb_vars)) ; combination ++) {
    if (nb_bits(combination) == Eqs->NbRows) {
      k=0;
      // 1- put the m colums in a square matrix
      for (j=0; j< nb_vars; j++) {
	if ((combination>>j)%2) {
	  for (i=0; i< Eqs->NbRows; i++) {
	    value_assign(Square_Mat->p[i][k], Eqs->p[i][j]);
	  }
	  k++;
	}
      }
      // 2- see if the matrix is full-row-rank
      right_hermite(Square_Mat, &Q, &H, &U);
      Matrix_Free(Q);
      Matrix_Free(U);
      // if it is full-row-rank, we have found a set of variables that can be eliminated. (to be prooved in order to be clean)
      if (H->p[Eqs->NbRows-1][Eqs->NbRows-1]!=0) {
	// 3- make the permutation matrix
	//  a- deal with the variables
	k=0;
	for (i=0; i< nb_vars; i++) {
	  // if the variable has to be eliminated, put them at the beginning
	  if (combination%2) {
	    permutation[i] = k;
	    k++;
	  }
	  // if not, put the variables at the end
	  else permutation[i] = Eqs->NbRows+nb_parms+ i-k;
	  combination>>=1;
	}
	//  b- deal with the parameters
	for (i=0; i< nb_parms; i++) {
	  permutation[nb_vars+i] = Eqs->NbRows+i;
	}
	//  c- deal with the constant
	permutation[Eqs->NbColumns-2] = Eqs->NbColumns-2;
	// have a look at the permutation
	// printf(" Permutation : ");
	// for (i=0; i< Eqs->NbColumns-1; i++) printf("%u ", permutation[i]);
	// return it.
	return permutation;
      }
      Matrix_Free(H);
    }
  }
  // if no combination of variables allow an elimination, then return an error code.
  return NULL;
}
// find_a_permutation



// compute the permutation of variables and parameters, according to some variables to keep.
// put the variables not to be kept at the beginning, then the parameters and finally the variables to be kept.
// strongly related to the function compress_to_full_dim2
unsigned int * permutation_for_full_dim2(unsigned int * vars_to_keep, unsigned int nb_keep, unsigned int nb_vars_parms, unsigned int nb_parms) {
  unsigned int * permutation = (unsigned int*)malloc(sizeof(unsigned int) * nb_vars_parms+1);
  unsigned int i;
  int cur_keep =0, cur_go = 0; // current number of variables to eliminate and to keep
  for (i=0; i< nb_vars_parms - nb_parms; i++) {
    if (i==vars_to_keep[cur_keep]) {
      permutation[i] = nb_vars_parms-nb_keep+cur_keep;
      cur_keep++;
    }
    else {
      permutation[i] = cur_go;
      cur_go++;
    }
  }
  // parameters are just left-shifted
  for (i=0; i< nb_parms; i++)
    permutation[i+nb_vars_parms-nb_parms] = i+nb_vars_parms-nb_parms-nb_keep;

  // contants stay where they are
  permutation[nb_vars_parms] = nb_vars_parms;
  return permutation;
} // permutation_for_full_dim2


// END OF STUFF WITH PERMUTATIONS
