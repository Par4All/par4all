/* Legal characters to start a comment line */
#define START_COMMENT_LINE "CcDd*!"

/* definition of extern variables. see comments and declarations in file
parser.c */

extern char *LogFN;
extern FILE *LogFD;
extern char *ModulesFN;
extern char *CurrentFN;
extern int debugging_level;
/* extern entity CurrentFunction; */
extern char *CurrentPackage;
extern entity DynamicArea;
extern entity StaticArea;
extern cons * FormalParameters;
extern char lab_I[];
extern char FormatValue[];
extern FILE * syn_in; /* the file read in by the scanner */
extern int line_b_I, line_e_I, line_b_C, line_e_C;
extern char Comm[]; /* the current comment */
extern char PrevComm[]; /* the previous comment */
extern int iComm;
extern int iPrevComm;


/* definition of implementation dependent constants */

#include "constants.h"
#define HASH_SIZE 1013
#define FORMATLENGTH (4096)
#define LOCAL static

#define abs(v) (((v) < 0) ? -(v) : (v))

extern char * getenv();



#define Warning(f,m) \
(user_warning(f,"Warning between lines %d and %d\n%s\n",line_b_I,line_e_I,m) )

#define FatalError(f,m) \
(pips_error(f,"Fatal error between lines %d and %d\n%s\n",line_b_I,line_e_I,m))
