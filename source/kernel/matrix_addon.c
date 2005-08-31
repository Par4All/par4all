// Polylib Matrix addons
// Mainly deals with polyhedra represented as a matrix (implicit form)
// B. Meister
#include<polylib/polylib.h>
#include <polylib/matrix_addon.h>

// splits a matrix of constraints M into a matrix of equalities Eqs and a matrix of inequalities Ineqs
// allocs the new matrices.
void split_constraints(Matrix const * M, Matrix ** Eqs, Matrix **Ineqs) {
  unsigned int i, j, k_eq, k_ineq, nb_eqs=0;

  // 1- count the number of equations;
  for (i=0; i< M->NbRows; i++)     
    if (value_zero_p(M->p[i][0])) nb_eqs++;

  // 2- extract the two matrices of equations
  (*Eqs) = Matrix_Alloc(nb_eqs, M->NbColumns);
  (*Ineqs) = Matrix_Alloc(M->NbRows-nb_eqs, M->NbColumns);

  k_eq = k_ineq = 0;
  for(i=0; i< M->NbRows; i++) {
    if (value_zero_p(M->p[i][0])) 
      {
	for(j=0; j< M->NbColumns; j++)
	  value_assign((*Eqs)->p[k_eq][j], M->p[i][j]);
	k_eq++;
      }
    else
       {
	for(j=0; j< M->NbColumns; j++)
	  value_assign((*Ineqs)->p[k_ineq][j], M->p[i][j]);
	k_ineq++;
      }
  }
}

// returns the dim-dimensional identity matrix
Matrix * Identity_Matrix(unsigned int dim) {
  Matrix * ret = Matrix_Alloc(dim, dim);
  unsigned int i,j;
  for (i=0; i< dim; i++) {
    for (j=0; j< dim; j++) {
      if (i==j) 
	{ value_set_si(ret->p[i][j], 1); } 
      else value_set_si(ret->p[i][j], 0);
    }
  }
  return ret;
} // Identity_Matrix

// given a n x n integer transformation matrix transf, compute its inverse M/g, where M is a nxn integer matrix.
// g is a common denominator for elements of (transf^{-1})
void mtransformation_inverse(Matrix * transf, Matrix ** inverse, Value * g) {
  Value factor;
  unsigned int i,j;

  value_init(*g);
  value_set_si(*g,1);

  // a - compute the inverse as usual (n x (n+1) matrix)
  Matrix * tmp = Matrix_Copy(transf);
  Matrix * inv = Matrix_Alloc(transf->NbRows, transf->NbColumns+1);
  MatInverse(tmp, inv);
  Matrix_Free(tmp);

  // b - as it is rational, put it to the same denominator
  (*inverse) = Matrix_Alloc(transf->NbRows, transf->NbRows);
  for (i=0; i< inv->NbRows; i++) 
    Lcm3(*g, inv->p[i][inv->NbColumns-1],g);
  for (i=0; i< inv->NbRows; i++) {
    value_division(factor, *g, inv->p[i][inv->NbColumns-1]);
    for (j=0; j< (*inverse)->NbColumns; j++) value_multiply((*inverse)->p[i][j], inv->p[i][j],  factor);
  }

  // c- clean up
  value_clear(factor);
  Matrix_Free(inv);
} // mtransformation_inverse


// takes a transformation matrix, and expands it to a higher dimension with the identity matrix 
// regardless of it homogeneousness
Matrix * mtransformation_expand_left_to_dim(Matrix * M, int new_dim) {
  assert(new_dim>=M->NbColumns);
  assert(M->NbRows==M->NbColumns);
  Matrix * ret = Identity_Matrix(new_dim);
  int offset = new_dim-M->NbRows;
  unsigned int i,j;
  for (i=0; i< M->NbRows; i++)
    for (j=0; j< M->NbRows; j++)
      value_assign(ret->p[offset+i][offset+j], M->p[i][j]);
  return ret;
} // mtransformation_expand_left_to_dim


// simplify a matrix seen as a polyhedron, by dividing its rows by the gcd of their elements.
void mpolyhedron_simplify(Matrix * polyh) {
  int i, j;
  Value cur_gcd;
  value_init(cur_gcd);
  for (i=0; i< polyh->NbRows; i++) {
    value_set_si(cur_gcd, 0);
    for (j=1; j< polyh->NbColumns; j++) Gcd(cur_gcd, polyh->p[i][j], &cur_gcd);
    printf(" gcd[%d] = ", i); value_print(stdout, VALUE_FMT, cur_gcd);printf("\n");
    for (j=1; j< polyh->NbColumns; j++) value_division(polyh->p[i][j], polyh->p[i][j], cur_gcd);
  }
  value_clear(cur_gcd);
} // mpolyhedron_simplify


// inflates a polyhedron (represented as a matrix) P, so that the apx of its Ehrhart Polynomial is an upper bound of the Ehrhart polynomial of P
// WARNING: this inflation is supposed to be applied on full-dimensional polyhedra.
void mpolyhedron_inflate(Matrix * polyh, unsigned int nb_parms) {
  unsigned int i,j;
  unsigned nb_vars = polyh->NbColumns-nb_parms-2;
  Value infl;
  value_init(infl);
  // substract the sum of the negative coefficients of each inequality
  for (i=0; i< polyh->NbRows; i++) {
    value_set_si(infl, 0);
    for (j=0; j< nb_vars; j++) {
      if (value_sign(polyh->p[i][j])<0)
	value_addto(infl, infl, polyh->p[i][j]);
    }
    // here, we substract a negative value
    value_subtract(polyh->p[i][polyh->NbColumns-1], polyh->p[i][polyh->NbColumns-1], infl);
  }
  value_clear(infl);
} // mpolyhedron_inflate


// deflates a polyhedron (represented as a matrix) P, so that the apx of its Ehrhart Polynomial is a lower bound of the Ehrhart polynomial of P
// WARNING: this deflation is supposed to be applied on full-dimensional polyhedra.
void mpolyhedron_deflate(Matrix * polyh, unsigned int nb_parms) {
  unsigned int i,j;
  unsigned nb_vars = polyh->NbColumns-nb_parms-2;
  Value defl;
  value_init(defl);
  // substract the sum of the negative coefficients of each inequality
  for (i=0; i< polyh->NbRows; i++) {
    value_set_si(defl, 0);
    for (j=0; j< nb_vars; j++) {
      if (value_sign(polyh->p[i][j])>0)
	value_addto(defl, defl, polyh->p[i][j]);
    }
    // here, we substract a negative value
    value_subtract(polyh->p[i][polyh->NbColumns-1], polyh->p[i][polyh->NbColumns-1], defl);
  }
  value_clear(defl);
} // mpolyhedron_deflate


// use an eliminator row to eliminate a variable in a victim row (without changing the sign of the victim row -> important if it is an inequality).
void eliminate_var_with_constr(Matrix * Eliminator, unsigned int eliminator_row, Matrix * Victim, unsigned int victim_row, unsigned int var_to_elim) {
  Value cur_lcm, mul_a, mul_b, a, b, sb;
  Value tmp, tmp2;
  int k; 
  value_init(cur_lcm); value_init(mul_a); value_init(mul_b); value_init(a); value_init(b); value_init(sb);
  value_init(tmp); value_init(tmp2);
  // if the victim coefficient is not zero 
  if (value_notzero_p(Victim->p[victim_row][var_to_elim+1])) {
    value_assign(a, Eliminator->p[eliminator_row][var_to_elim+1]);
    value_assign(b, Victim->p[victim_row][var_to_elim+1]);
    Lcm3(a, b, &cur_lcm);
    // multiplication factor for the current constraint
    value_division(tmp, cur_lcm, b);
    value_absolute(mul_a, tmp); // IT HAS TO BE POSITIVE (otherwise you may modify the sign of your constraint)
    value_absolute(tmp, b);
    value_division(sb, tmp, b); // sb represents the sign of b
    // multiplication factor for the constraint to project
#ifdef GNUMP
    mpz_divexact(tmp, cur_lcm, a);
    value_multiply(mul_b, tmp, sb);
#else
    mul_b = cur_lcm / a * sb;
#endif

    value_assign(Victim->p[victim_row][0], Victim->p[victim_row][0]);
    for (k=1; k<Victim->NbColumns; k++) {
      value_multiply(tmp, Victim->p[victim_row][k], mul_a);
      value_multiply(tmp2, Eliminator->p[eliminator_row][k], mul_b);
      value_subtract(Victim->p[victim_row][k], tmp, tmp2);
      // = Victim->p[victim_row][k] * mul_a -  Eliminator->p[eliminator_row][k] * mul_b;
    }
  }
  value_clear(cur_lcm); value_clear(mul_a); value_clear(mul_b); value_clear(a); value_clear(b); value_clear(sb);
  value_clear(tmp); value_clear(tmp2);
}
// eliminate_var_with_constr


// STUFF WITH PARTIAL MAPPINGS (Mappings to a subset of the variables/parameters) : on the first or last variables/parameters

// compress the last vars/pars of the polyhedron M expressed as a polylib matrix
// - adresses the full-rank compressions only
// - modfies M
void mpolyhedron_compress_last_vars(Matrix * M, Matrix * compression) {
  unsigned int i, j, k;
  unsigned int offset = M->NbColumns - compression->NbRows; // the computations on M will begin on column "offset"
  Matrix * M_tmp = Matrix_Alloc(1, M->NbColumns);
  assert(compression->NbRows==compression->NbColumns);
  // basic matrix multiplication (using a temporary row instead of a whole temporary matrix), but with a column offset
  for(i=0; i< M->NbRows; i++) {
    for (j=0; j< compression->NbRows; j++) {
      value_set_si(M_tmp->p[0][j], 0);
      for (k=0; k< compression->NbRows; k++) {
	value_addmul(M_tmp->p[0][j], M->p[i][k+offset],compression->p[k][j] );
      }
    }
    for (j=0; j< compression->NbRows; j++) 
      value_assign(M->p[i][j+offset], M_tmp->p[0][j]);
  }
  Matrix_Free(M_tmp);
} // mpolyhedron_compress_last_vars


// use a set of m equalities Eqs to eliminate m variables in the polyhedron Ineqs represented as a matrix
// eliminates the m first variables
// - assumes that Eqs allow to eliminate the m equalities
//  -modifies Eqs and Ineqs
unsigned int mpolyhedron_eliminate_first_variables(Matrix * Eqs, Matrix * Ineqs) {
  unsigned int i, j, k;
  // eliminate one variable (index i) after each other
  for (i=0; i< Eqs->NbRows; i++) {
    // find j, the first (non-marked) row of Eqs with a non-zero coefficient
    for (j=0; j<Eqs->NbRows && (Eqs->p[j][i+1]==0 || ( !value_cmp_si(Eqs->p[j][0],2) )); j++);
    // if no row is found in Eqs that allows to eliminate variable i, return an error code (0)
    if (j==Eqs->NbRows) return 0;
    // else, eliminate variable i in Eqs and Ineqs with the j^th row of Eqs (and mark this row so we don't use it again for an elimination)
    for (k=j+1; k<Eqs->NbRows; k++)
      eliminate_var_with_constr(Eqs, j, Eqs, k, i);
    for (k=0; k< Ineqs->NbRows; k++)
      eliminate_var_with_constr(Eqs, j, Ineqs, k, i);
    // mark the row
    value_set_si(Eqs->p[j][0],2);
  }
  // un-mark all the rows
  for (i=0; i< Eqs->NbRows; i++) value_set_si(Eqs->p[i][0],0);
  return 1;
}

