/** 
 * $Id: testCompressParms.c,v 1.1 2006/03/13 06:43:49 loechner Exp $
 * 
 * Test routines for kernel/compress_parms.c functions
 * @author B. Meister, 3/2006
 * 
 */

#include <polylib/polylib.h>
#include <stdio.h>

#define maxRays 200


int main(int argc, char ** argv) {
  int isOk = 0;
  Matrix * A, * B;
  if (argc>1) {
    printf("Warning: No arguments taken into account: testing"
	   "remove_parm_eqs().\n");
  }

  A = Matrix_Read();
  B = Matrix_Read();
  if (isOk = test_Constraints_Remove_parm_eqs(A, B)) {
    printf("Constraints_Remove_parm_eqs() is ok for this example.\n");
  }
  else {
    printf("Constraints_Remove_parm_eqs() is NOT ok for this example.\n");
  }

  if (isOk = test_Polyhedron_Remove_parm_eqs(A, B)) {
    printf("Polyhedron_Remove_parm_eqs() is ok for this example.\n");
  }
  else {
    printf("Polyhedron_Remove_parm_eqs() is NOT ok for this example.\n");
  }
  Matrix_Free(A);
  Matrix_Free(B);

}


/** extracts the equalities involving the parameters only, try to introduce
    them back and compare the two polyhedra.
    Reads a polyhedron and a context.
 */
int test_Constraints_Remove_parm_eqs(Matrix * A, Matrix * B) {
  int isOk = 1;
  Matrix * M, *C, *Cp, * Eqs, *M1, *C1;
  Polyhedron *Pm, *Pc, *Pcp, *Peqs, *Pint;
  printf("----- test_Constraints_Remove_parm_eqs() -----\n");
  M1 = Matrix_Copy(A);
  C1 = Matrix_Copy(B);

  M = Matrix_Copy(M1);
  C = Matrix_Copy(C1);

   /* compute the combined polyhedron */
  Pm = Constraints2Polyhedron(M, maxRays);
  Pc = Constraints2Polyhedron(C, maxRays);
  Pcp = align_context(Pc, Pm->Dimension, maxRays);
  Polyhedron_Free(Pc);
  Pc = DomainIntersection(Pm, Pcp, maxRays);
  Polyhedron_Free(Pm);
  Polyhedron_Free(Pcp);
  Matrix_Free(M);
  Matrix_Free(C);

  /* extract the parm-equalities, expressed in the combined space */
  Eqs = Constraints_Remove_parm_eqs(&M1, &C1, 1);

  printf("Removed equalities: \n");
  show_matrix(Eqs); 
  printf("Polyhedron without equalities involving only parameters: \n");
  show_matrix(M1);  
  printf("Context without equalities: \n");
  show_matrix(C1);  
  
  /* compute the supposedly-same polyhedron, using the extracted equalities */
  Pm = Constraints2Polyhedron(M1, maxRays);
  Pcp = Constraints2Polyhedron(C1, maxRays);
  Peqs = align_context(Pcp, Pm->Dimension, maxRays);
  Polyhedron_Free(Pcp);
  Pcp = DomainIntersection(Pm, Peqs, maxRays);
  Polyhedron_Free(Peqs);
  Polyhedron_Free(Pm);
  Peqs = Constraints2Polyhedron(Eqs, maxRays);
  Matrix_Free(Eqs);
  Matrix_Free(M1);
  Matrix_Free(C1);
  Pint = DomainIntersection(Pcp, Peqs, maxRays);
  Polyhedron_Free(Pcp);
  Polyhedron_Free(Peqs);

  /* test their equality */
  if (!PolyhedronIncludes(Pint, Pc)) {
    isOk = 0;
  }
  else {
    if (!PolyhedronIncludes(Pc, Pint)) {
      isOk = 0;
    }
  }
  Polyhedron_Free(Pc);
  Polyhedron_Free(Pint);
  return isOk;
} /* test_Constraints_Remove_parm_eqs() */


/** extracts the equalities holding on the parameters only, try to introduce
    them back and compare the two polyhedra.
    Reads a polyhedron and a context.
 */
int test_Polyhedron_Remove_parm_eqs(Matrix * A, Matrix * B) {
  int isOk = 1;
  Matrix * M, *C;
  Polyhedron *Pm, *Pc, *Pcp, *Peqs, *Pint, *Pint1;
  printf("----- test_Polyhedron_Remove_parm_eqs() -----\n");

  M = Matrix_Copy(A);
  C = Matrix_Copy(B);

   /* compute the combined polyhedron */
  Pm = Constraints2Polyhedron(M, maxRays);
  Pc = Constraints2Polyhedron(C, maxRays);
  Pcp = align_context(Pc, Pm->Dimension, maxRays);
  Polyhedron_Free(Pc);
  Pint1 = DomainIntersection(Pm, Pcp, maxRays);
  Polyhedron_Free(Pm);
  Polyhedron_Free(Pcp);
  Matrix_Free(M);
  Matrix_Free(C);

  M = Matrix_Copy(A);
  C = Matrix_Copy(B);
  /* extract the parm-equalities, expressed in the combined space */
  Pm = Constraints2Polyhedron(M, maxRays);
  Pc = Constraints2Polyhedron(C, maxRays);
  Matrix_Free(M);
  Matrix_Free(C);
  Peqs = Polyhedron_Remove_parm_eqs(&Pm, &Pc, 1, 200);
  
  /* compute the supposedly-same polyhedron, using the extracted equalities */
  Pcp = align_context(Pc, Pm->Dimension, maxRays);
  Polyhedron_Free(Pc);
  Pc = DomainIntersection(Pm, Pcp, maxRays);
  Polyhedron_Free(Pm);
  Polyhedron_Free(Pcp);
 
  Pint = DomainIntersection(Pc, Peqs, maxRays);
  Polyhedron_Free(Pc);
  Polyhedron_Free(Peqs);

  /* test their equality */
  if (!PolyhedronIncludes(Pint, Pint1)) {
    isOk = 0;
  }
  else {
    if (!PolyhedronIncludes(Pint1, Pint)) {
      isOk = 0;
    }
  }
  Polyhedron_Free(Pint1);
  Polyhedron_Free(Pint);
  return isOk;
} /* test_remove_parm_eqs() */
