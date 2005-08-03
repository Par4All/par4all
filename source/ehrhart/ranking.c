// Tools to compute the ranking function of an iteration J : the number of integer points in P that are lexicographically inferior to J
// B. Meister 6/2005
// LSIIT-ICPS, UMR 7005 CNRS Université Louis Pasteur
// HiPEAC Network

#include <polylib/polylib.h>


// given the constraints of a polyhedron P (the matrix Constraints), returns the number of points that are lexicographically stricly lesser than a point I of P,
// defined by I = M.(J N 1)^T,
// where M is an integer matrix provided by the user.
// J are the n' first variables of the returned Ehrhart polynomial.
// If M is NULL, I = J is taken by default.
Enumeration *Ranking(Matrix * Constraints, Matrix * C, Matrix * M, 
		     unsigned MAXRAYS, char ** param_name) 
{
  unsigned i,j,k;
  unsigned nb_parms = C->NbColumns -2;
  unsigned nb_vars = Constraints->NbColumns-C->NbColumns;
  unsigned nb_new_parms;
  Enumeration * ranking;
  Matrix * cur_element, * C_times_J, * Klon;
  Polyhedron * P1, *C1;
  Polyhedron * lexico_lesser_union = NULL;

  if (M)
    assert(M->NbRows==nb_vars);

  if (M)
    nb_new_parms = M->NbColumns-C->NbColumns+1;
  else
    nb_new_parms = nb_vars;

  // the number of variables must be positive
  if (nb_vars<=0) {
    printf("\nRanking > No variables, returning NULL.\n"); 
    return NULL;
  }
  cur_element = Matrix_Alloc(Constraints->NbRows+nb_vars, 
			     Constraints->NbColumns+nb_new_parms);


  // 0- Put P in the first rows of cur_element
  for (i=0; i< Constraints->NbRows; i++) {
    for (j=0; j< nb_vars+1; j++)
      value_assign(cur_element->p[i][j], Constraints->p[i][j]);
    for (j=0; j< nb_parms+1; j++)
      value_assign(cur_element->p[i][j+nb_vars+nb_new_parms+1], 
		   Constraints->p[i][j+nb_vars+1]);
  }

  // 1- compute the Ehrhart polynomial of each disjoint polyhedron defining the lexicographic order
  for (k=0; k < nb_vars; k++) {

    // a- build the corresponding matrix
    // the nb of rows of cur_element is fake, so that we do not have to re-allocate it.
    cur_element->NbRows = Constraints->NbRows+k+1;

    // convert the last (strict) inequality into an equality
    if (k>=1) {
      value_set_si(cur_element->p[Constraints->NbRows+k-1][0], 0);
      value_increment(cur_element->p[Constraints->NbRows+k-1][cur_element->NbColumns-1], cur_element->p[Constraints->NbRows+k-1][cur_element->NbColumns-1]);
    }
    // build the k-th inequality form M
    value_set_si(cur_element->p[Constraints->NbRows+k][0], 1);
    value_set_si(cur_element->p[Constraints->NbRows+k][k+1], -1);
    if (M) {
      for (j=0; j< M->NbColumns; j++)
	value_assign(cur_element->p[Constraints->NbRows+k][j+nb_vars+1], M->p[k][j]);
    }
    else
      value_set_si(cur_element->p[Constraints->NbRows+k][nb_vars+k+1], 1);
    value_decrement(cur_element->p[Constraints->NbRows+k][cur_element->NbColumns-1], cur_element->p[Constraints->NbRows+k][cur_element->NbColumns-1]); // we want a strict inequality
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
  C_times_J = Matrix_Alloc(C->NbRows+Constraints->NbRows, C->NbColumns+nb_new_parms);
  // copy the initial context while adding the new parameters
  for (i=0; i< C->NbRows; i++) {
    value_assign(C_times_J->p[i][0], C->p[i][0]);
    for (j=0; j< nb_parms; j++)
      value_assign(C_times_J->p[i][j+nb_vars+1], C->p[i][j+1]); 
  }

  // add the constraints PM(J N 1)^T >=0
  if (M) {
    for (i=0; i< Constraints->NbRows; i++) {
      value_assign(C_times_J->p[i+C->NbRows][0], Constraints->p[i][0]);
      for (j=0; j< nb_new_parms; j++) {
	value_set_si(C_times_J->p[i+C->NbRows][j+1], 0);
	for (k=0; k< nb_vars; k++) {
	  value_addmul(C_times_J->p[i+C->NbRows][j+1], Constraints->p[i][k+1], M->p[k][j]);
	}
      }
      for (j=0; j< nb_parms+1; j++) {
	value_set_si(C_times_J->p[i+C->NbRows][j+nb_new_parms+1], 0);
	for (k=0; k< nb_vars; k++) {
	  value_addmul(C_times_J->p[i+C->NbRows][j+nb_new_parms+1], Constraints->p[i][k+1], M->p[k][j+nb_new_parms]);
	}
	value_addto(C_times_J->p[i+C->NbRows][j+nb_new_parms+1], C_times_J->p[i+C->NbRows][j+nb_new_parms+1], Constraints->p[i][j+nb_vars+1]);
      }
    }
  }
  else { // if M=NULL, just use I = J => copy Constraints into the new context
    for (i=0; i< Constraints->NbRows; i++) {
      for (j=0; j< Constraints->NbColumns; j++) {
	value_assign(C_times_J->p[i+C->NbRows][j], Constraints->p[i][j]);
      }
    }
  }
  show_matrix(C_times_J);
  C1 = Constraints2Polyhedron(C_times_J, MAXRAYS);

  // 3- Compute the ranking, which is the sum of the Ehrhart polynomials of the n disjoint polyhedra we just put in P1.
  // OPT : our polyhdera are (already) disjoint, so Domain_Enumerate does probably too much work uselessly
  ranking = Domain_Enumerate(P1, C1, MAXRAYS, param_name);

  // 4- clean up
  Polyhedron_Free(P1);
  Polyhedron_Free(C1);
  Matrix_Free(cur_element);
  Matrix_Free(C_times_J);
  return ranking;
} // Ranking
