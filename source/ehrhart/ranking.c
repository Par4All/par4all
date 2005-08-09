// Tools to compute the ranking function of an iteration J : the number of integer points in P that are lexicographically inferior to J
// B. Meister 6/2005
// LSIIT-ICPS, UMR 7005 CNRS Université Louis Pasteur
// HiPEAC Network

#include <polylib/polylib.h>
#include <polylib/ranking.h>

/*
 * Returns the number of points in P that are lexicographically
 * smaller than a given point in D.
 * When P == D, this is the conventional ranking function.
 * P and D are assumed to have the same parameter domain C.
 * The variables in the Enumeration correspond to the variables
 * in D followed by the parameter of D (the variables of C).
 */
Enumeration *Ranking(Polyhedron *P, Polyhedron *D, Polyhedron *C, 
		     unsigned MAXRAYS) 
{
  unsigned i, j, k, r;
  unsigned nb_parms = C->Dimension;
  unsigned nb_vars = P->Dimension - C->Dimension;
  unsigned nb_new_parms;
  Enumeration * ranking;
  Matrix * cur_element, * C_times_J, * Klon;
  Polyhedron * P1, *C1;
  Polyhedron * lexico_lesser_union = NULL;

  POL_ENSURE_INEQUALITIES(C);
  POL_ENSURE_INEQUALITIES(D);
  POL_ENSURE_INEQUALITIES(P);

  assert(P->Dimension == D->Dimension);
  nb_new_parms = nb_vars;

  // the number of variables must be positive
  if (nb_vars<=0) {
    printf("\nRanking > No variables, returning NULL.\n"); 
    return NULL;
  }
  cur_element = Matrix_Alloc(P->NbConstraints+nb_vars, 
			     P->Dimension+nb_new_parms+2);


  // 0- Put P in the first rows of cur_element
  for (i=0; i < P->NbConstraints; i++) {
    Vector_Copy(P->Constraint[i], cur_element->p[i], nb_vars+1);
    Vector_Copy(P->Constraint[i]+1+nb_vars, 
		cur_element->p[i]+1+nb_vars+nb_new_parms, nb_parms+1);
  }

  // 1- compute the Ehrhart polynomial of each disjoint polyhedron defining the lexicographic order
  for (k=0, r = P->NbConstraints; k < nb_vars; k++, r++) {

    // a- build the corresponding matrix
    // the nb of rows of cur_element is fake, so that we do not have to re-allocate it.
    cur_element->NbRows = r+1;

    // convert the previous (strict) inequality into an equality
    if (k>=1) {
      value_set_si(cur_element->p[r-1][0], 0);
      value_set_si(cur_element->p[r][cur_element->NbColumns-1], 0);
    }
    // build the k-th inequality from P
    value_set_si(cur_element->p[r][0], 1);
    value_set_si(cur_element->p[r][k+1], -1);
    value_set_si(cur_element->p[r][nb_vars+k+1], 1);
    // we want a strict inequality
    value_set_si(cur_element->p[r][cur_element->NbColumns-1], -1);
    show_matrix(cur_element);

    // b- add it to the current union
    // as Constraints2Polyhedron modifies its input, we must clone cur_element
    Klon = Matrix_Copy(cur_element);
    P1 = Constraints2Polyhedron(Klon, MAXRAYS);
    Matrix_Free(Klon);
    P1->next = lexico_lesser_union;
    lexico_lesser_union = P1;
  }
  
  // 2- as we introduce n parameters, we must introduce them into the context as well
  // The added constraints are P.M.(J N 1 )^T >=0
  C_times_J = Matrix_Alloc(C->NbConstraints + D->NbConstraints, D->Dimension+2);
  // copy the initial context while adding the new parameters
  for (i = 0; i < C->NbConstraints; i++) {
    value_assign(C_times_J->p[i][0], C->Constraint[i][0]);
    Vector_Copy(C->Constraint[i]+1, C_times_J->p[i]+1+nb_new_parms, nb_parms+1);
  }

  /* copy constraints from evaluation domain */
  for (i = 0; i < D->NbConstraints; i++)
    Vector_Copy(D->Constraint[i], C_times_J->p[C->NbConstraints+i], D->Dimension+2);

  show_matrix(C_times_J);
  C1 = Constraints2Polyhedron(C_times_J, POL_NO_DUAL);

  // 3- Compute the ranking, which is the sum of the Ehrhart polynomials of the n disjoint polyhedra we just put in P1.
  // OPT : our polyhdera are (already) disjoint, so Domain_Enumerate does probably too much work uselessly
  ranking = Domain_Enumerate(P1, C1, MAXRAYS, NULL);

  // 4- clean up
  Domain_Free(P1);
  Polyhedron_Free(C1);
  Matrix_Free(cur_element);
  Matrix_Free(C_times_J);
  return ranking;
} // Ranking
