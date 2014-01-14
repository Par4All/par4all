/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#define INFAISABLE 0
#define FAISABLE 1

/* maximun number of nested loops */
#define MAXDEPTH 9
/* maximum number of scalar variables */
#define MAXSV 1024


/*the variables for the statistics of test of dependence and parallelization */
extern int NbrArrayDepInit;
extern int NbrIndepFind;
extern int NbrAllEquals;
extern int NbrDepCnst;
extern int NbrDepExact;
extern int NbrDepInexactEq;
extern int NbrDepInexactFM;
extern int NbrDepInexactEFM;
extern int NbrScalDep;
extern int NbrIndexDep;
extern int deptype[5][3], constdep[5][3];
extern int NbrTestCnst;
extern int NbrTestGcd;
extern int NbrTestSimple; /* by sc_normalize() */
extern int NbrTestDiCnst;
extern int NbrTestProjEqDi;
extern int NbrTestProjFMDi;
extern int NbrTestProjEq;
extern int NbrTestProjFM;
extern int NbrTestDiVar;
extern int NbrProjFMTotal;
extern int NbrFMSystNonAug;
extern int FMComp[18]; 
extern bool is_test_exact;
extern bool is_test_inexact_eq;
extern bool is_test_inexact_fm;
extern bool is_dep_cnst;
extern bool is_test_Di;
extern bool Finds2s1;

extern int Nbrdo;

/* Definition for the dependance_verticies_p function
 */

#define FLOW_DEPENDANCE 1
#define ANTI_DEPENDANCE 2
#define OUTPUT_DEPENDANCE 4
#define INPUT_DEPENDANCE 8






