#define INFAISABLE 0
#define FAISABLE 1

/* maximun number of nested loops */
#define MAXDEPTH 9
#define MAXSV 100


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
extern int FMComp[17]; 
extern boolean is_test_exact;
extern boolean is_test_inexact_eq;
extern boolean is_test_inexact_fm;
extern boolean is_dep_cnst;
extern boolean is_test_Di;
extern boolean Finds2s1;

extern int Nbrdo;

/* Definition for the dependance_verticies_p function
 */

#define FLOW_DEPENDANCE 1
#define ANTI_DEPENDANCE 2
#define OUTPUT_DEPENDANCE 4
#define INPUT_DEPENDANCE 8






