/* comp_util.c
 *
 * useful routines for evaluation of the complexity of a program
 *
 * boolean complexity_check(comp)
 * void complexity_check_and_warn(function_name, comp)
 * void good_complexity_assert(function_name, comp)
 * void complexity_fprint(fd, comp, print_stats_p, print_local_names_p)
 * char *complexity_sprint(comp, print_stats_p, print_local_names_p)
 * void fprint_statement_complexity(module, stat, hash_statement_to_complexity)
 * void prc(comp) (for dbx)
 * void prp(pp)   (for dbx)
 * void prv(pv)   (for dbx)
 * void fprint_cost_table(fd)
 * void init_cost_table();
 * int intrinsic_cost(name, argstype)
 * boolean is_inferior_basic(basic1, basic2)
 * basic simple_basic_dup(b)
 * float constant_entity_to_float(e)
 * void trace_on(va_alist)
 * void trace_off()
 * list entity_list_reverse(l)
 * boolean is_linear_unstructured(unstr)
 * void add_formal_parameters_to_hash_table(mod, hash_complexity_params)
 * void remove_formal_parameters_from_hash_table(mod, hash_complexity_params)
 * hash_table fetch_callees_complexities(module_name)
 * hash_table fetch_complexity_parameters(module_name)
 */
/* Modif:
  -- entity_local_name is replaced by module_local_name. LZ 230993
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>      /* getenv */
#include <stdarg.h>

#include "linear.h"

#include "genC.h"
#include "database.h"
#include "ri.h"
#include "complexity_ri.h"
#include "resources.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "text-util.h"     /* print_text */
#include "effects-generic.h"
#include "misc.h"
#include "constants.h"     /* IMPLIED_DO_NAME is defined there */
#include "properties.h"    /* get_string_property is defined there */
#include "matrice.h"
#include "polynome.h"
#include "complexity.h"

/* for debugging */
#define INDENT_BLANKS "  "
#define INDENT_VLINE  "| "
#define INDENT_BACK   "-"
#define INDENT_INTERVAL 2

/* return TRUE if allright */
boolean complexity_check(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_error("complexity_check", "complexity undefined");

    if ( !complexity_zero_p(comp) ) {
	return (polynome_check(complexity_polynome(comp)));
    }
    return (TRUE);
}

void complexity_check_and_warn(s,comp)
char *s;
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_error("complexity_check_and_warn", "complexity undefined");

    if ( complexity_zero_p(comp) ) {
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr,"complexity ZERO for %s\n",s);
	}
    }	
    if (!complexity_check(comp))
	user_warning(s,"Bad internal complexity representation!\n");
}    

void good_complexity_assert(char * function, complexity comp)
{
    if (!complexity_check(comp))
	pips_error(function, "bad internal complexity representation\n");
}

/* duplicates complexity comp */
complexity complexity_dup(comp)
complexity comp;
{
    if ( COMPLEXITY_UNDEFINED_P(comp) ) 
	pips_error("complexity_dup", "complexity undefined");

    if ( complexity_zero_p(comp) ) 
	return (make_zero_complexity());
    else {
	varcount   vc = complexity_varcount(comp);
	rangecount rc = complexity_rangecount(comp);
	ifcount    ic = complexity_ifcount(comp);

	varcount newvc = make_varcount(varcount_symbolic(vc), 
				       varcount_guessed(vc),
				       varcount_bounded(vc), 
				       varcount_unknown(vc));
	rangecount newrc = make_rangecount(rangecount_profiled(rc), 
					   rangecount_guessed(rc),
					   rangecount_bounded(rc),
					   rangecount_unknown(rc));
	ifcount newic = make_ifcount(ifcount_profiled(ic), 
				     ifcount_computed(ic),
				     ifcount_halfhalf(ic));
    
	Ppolynome ppdup = polynome_dup(complexity_polynome(comp));
	complexity compl = make_complexity(ppdup, newvc, newrc, newic);

	return(compl);
    }
}

/* remove complexity comp */
void complexity_rm(pcomp)
complexity *pcomp;
{
    if ( COMPLEXITY_UNDEFINED_P(*pcomp) )
	pips_error("complexity_rm", "undefined complexity\n");

    if ( !complexity_zero_p(*pcomp) ) 
	free_complexity(*pcomp);
    *pcomp = complexity_undefined;
}

char *complexity_sprint(comp, print_stats_p, print_local_names_p)
complexity comp;
boolean print_stats_p, print_local_names_p;
{
#define COMPLEXITY_BUFFER_SIZE 1024
    static char t[COMPLEXITY_BUFFER_SIZE];
    char *s, *p;
    extern boolean is_inferior_pvarval(Pvecteur *, Pvecteur *);

    s = t;

    if ( COMPLEXITY_UNDEFINED_P(comp) )
	pips_error("complexity_sprint", "complexity undefined\n");
    else {
	varcount vc   = complexity_varcount(comp);
	rangecount rc = complexity_rangecount(comp);
	ifcount ic    = complexity_ifcount(comp);

	if ( print_stats_p ) {
	    sprintf(s, "[(var:%d/%d/%d/%d)", varcount_symbolic(vc),
		                             varcount_guessed(vc),
		                             varcount_bounded(vc), 
		                             varcount_unknown(vc));
	    sprintf(s+strlen(s), " (rng:%d/%d/%d/%d)", 
		                             rangecount_profiled(rc),
		                             rangecount_guessed(rc),
		                             rangecount_bounded(rc), 
		                             rangecount_unknown(rc));

	    sprintf(s+strlen(s), " (ifs:%d/%d/%d)]  ", 
		                             ifcount_profiled(ic), 
		                             ifcount_computed(ic),
		                             ifcount_halfhalf(ic));


	    s = strchr(s, '\0');
	}
	p = polynome_sprint(complexity_polynome(comp),
			    (print_local_names_p ? variable_local_name 
			                         : variable_name),
			    is_inferior_pvarval);
	strcpy(s, p);
    }
    pips_assert("complexity_sprint", strlen(s) < COMPLEXITY_BUFFER_SIZE);
    return((char *) strdup((char *) t));
}

void complexity_fprint(fd, comp, print_stats_p, print_local_names_p)
FILE *fd;
complexity comp;
boolean print_stats_p, print_local_names_p;
{
    char *s = complexity_sprint(comp, print_stats_p, print_local_names_p);

    fprintf(fd, "%s\n", s);
    free(s);
}

void complexity_dump(complexity comp)
{
    complexity_fprint(stderr, comp, FALSE, TRUE);
}

void prc(comp)   /* for dbxtool: "print complexity" */
complexity comp;
{
    complexity_fprint(stderr, comp, TRUE, TRUE);
}

void prp(pp)     /* for dbxtool: "print polynome" */
Ppolynome pp;
{
    polynome_fprint(stderr, pp, variable_name, is_inferior_pvarval);
    fprintf(stderr, "\n");
}

void prv(pv)     /* for dbxtool: "print vecteur (as a monome)" */
Pvecteur pv;
{
    vect_fprint_as_monome(stderr, pv, BASE_NULLE, variable_name, ".");
    fprintf(stderr, "\n");
}

void fprint_statement_complexity(module, stat, hash_statement_to_complexity)
entity module;
statement stat;
hash_table hash_statement_to_complexity;
{
    text t = text_statement(module, 0, stat);
    complexity comp;

    comp = ((complexity) hash_get(hash_statement_to_complexity,(char *)stat));
    if (COMPLEXITY_UNDEFINED_P(comp))
	pips_error("fprint_statement_complexity","undefined complexity\n");
    else {
	fprintf(stderr, "C -- ");
	complexity_fprint(stderr, comp, DO_PRINT_STATS, PRINT_LOCAL_NAMES);
    }
    print_text(stderr, t);
}

/* The table intrinsic_cost_table[] gathers cost information
 * of each intrinsic's cost; those costs are dynamically loaded
 * from user files. It also returns the "minimum" type
 * of the result of each intrinsic,
 * specified by its basic_tag and number of memory bytes.
 * ("bigger" and "minimum" refer to the order relation
 * defined in the routine "is_inferior_basic"; the tag
 * is_basic_overloaded is used as a don't care tag)
 * (ex: SIN has a type of FLOAT even if its arg is an INT)
 *
 * Modif:
 *  -- LOOP_OVERHEAD and CALL_OVERHEAD are added, 280993 LZ
 *  -- LOOP_OVERHEAD is divided into two: INIT and BRAANCH 081093 LZ
 */
struct intrinsic_cost_rec {
    char *name;
    int min_basic_result;
    int min_nbytes_result;
    int int_cost;
    int float_cost;
    int double_cost;
    int complex_cost;
    int dcomplex_cost;
} intrinsic_cost_table[] = {

    { "+",                        is_basic_int, INT_NBYTES, EMPTY_COST },
    { "-",                        is_basic_int, INT_NBYTES, EMPTY_COST },
    { "*",                        is_basic_int, INT_NBYTES, EMPTY_COST },
    { "/",                        is_basic_int, INT_NBYTES, EMPTY_COST },
    { "--",                       is_basic_int, INT_NBYTES, EMPTY_COST },
    { "**",                       is_basic_int, INT_NBYTES, EMPTY_COST },

    { LOOP_INIT_OVERHEAD,         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { LOOP_BRANCH_OVERHEAD,       is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CONDITION_OVERHEAD,         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { CALL_ZERO_OVERHEAD,         is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_ONE_OVERHEAD,  	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_TWO_OVERHEAD,  	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_THREE_OVERHEAD,	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_FOUR_OVERHEAD, 	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_FIVE_OVERHEAD, 	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_SIX_OVERHEAD,  	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { CALL_SEVEN_OVERHEAD,	  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { ONE_INDEX_NAME,             is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { TWO_INDEX_NAME,             is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { THREE_INDEX_NAME,           is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { FOUR_INDEX_NAME,            is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { FIVE_INDEX_NAME,            is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { SIX_INDEX_NAME,             is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { SEVEN_INDEX_NAME,           is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { MEMORY_READ_NAME,           is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "=",                        is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { ".EQV.",                    is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".NEQV.",                   is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".OR.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".AND.",                    is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".LT.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".GT.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".LE.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".GE.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".EQ.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { ".NE.",                     is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { "//",                       is_basic_string,  ZERO_BYTE, EMPTY_COST },
    { ".NOT.",                    is_basic_logical, ZERO_BYTE, EMPTY_COST },

    { "CONTINUE",                 is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "ENDDO",                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "RETURN",                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "STOP",                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "END",                      is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "FORMAT",                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { "INT",                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "IFIX",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "IDINT",                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "REAL",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "FLOAT",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "SNGL",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DBLE",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CMPLX",                    is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "ICHAR",                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "CHAR",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "AINT",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "DINT",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ANINT",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DNINT",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "NINT",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "IDNINT",                   is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "IABS",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "ABS",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DABS",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CABS",                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },

    { "MOD",                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "AMOD",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DMOD",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ISIGN",                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "SIGN",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DSIGN",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "IDIM",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "DIM",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DDIM",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "DPROD",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "MAX",                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "MAX0",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "AMAX1",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DMAX1",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "AMAX0",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "MAX1",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "MIN",                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "MIN0",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "AMIN1",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DMIN1",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "AMIN0",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "MIN1",                     is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "LEN",                      is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "INDEX",                    is_basic_int,     INT_NBYTES,     EMPTY_COST },
    { "AIMAG",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "CONJG",                    is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "SQRT",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DSQRT",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CSQRT",                    is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },

    { "EXP",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DEXP",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CEXP",                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "ALOG",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DLOG",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CLOG",                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "LOG",                      is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "ALOG10",                   is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DLOG10",                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "LOG10",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "SIN",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DSIN",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CSIN",                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "COS",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DCOS",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "CCOS",                     is_basic_complex, COMPLEX_NBYTES, EMPTY_COST },
    { "TAN",                      is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DTAN",                     is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ASIN",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DASIN",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ACOS",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DACOS",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ATAN",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DATAN",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "ATAN2",                    is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DATAN2",                   is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "SINH",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DSINH",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "COSH",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DCOSH",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },
    { "TANH",                     is_basic_float,   FLOAT_NBYTES,   EMPTY_COST },
    { "DTANH",                    is_basic_float,   DOUBLE_NBYTES,  EMPTY_COST },

    { "LGE",                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { "LGT",                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { "LLE",                      is_basic_logical, ZERO_BYTE, EMPTY_COST },
    { "LLT",                      is_basic_logical, ZERO_BYTE, EMPTY_COST },

    { LIST_DIRECTED_FORMAT_NAME,  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { UNBOUNDED_DIMENSION_NAME,   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { "WRITE",                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "REWIND",                   is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "OPEN",                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "CLOSE",                    is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "READ",                     is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "BUFFERIN",                 is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "BUFFEROUT",                is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { "ENDFILE",                  is_basic_overloaded, ZERO_BYTE, EMPTY_COST },
    { IMPLIED_DO_NAME,            is_basic_overloaded, ZERO_BYTE, EMPTY_COST },

    { NULL,                         0, ZERO_BYTE, EMPTY_COST },
};


void fprint_cost_table(fd)
FILE *fd;
{
    struct intrinsic_cost_rec *p = intrinsic_cost_table;
    boolean skip_one_line = FALSE;
    
    fprintf(fd, "\nIntrinsic cost table:\n\n");
    fprintf(fd, "        Intrinsic name        int    float   double   complex   dcomplex\n");
    fprintf(fd, "------------------------------------------------------------------------\n");
	    
    for(; p->name != NULL ;p++) {
	if (1 ||(p->int_cost      != 0) ||
	    (p->float_cost    != 0) ||
	    (p->double_cost   != 0) ||
	    (p->complex_cost  != 0) ||
	    (p->dcomplex_cost != 0)) {
	    if (skip_one_line) {
		fprintf(fd, "%25s|\n", "");
		skip_one_line = FALSE;
	    }
	    fprintf(fd, "%22.21s   |%6d %6d %7d %8d %8d\n",
		    p->name, p->int_cost, p->float_cost,
		    p->double_cost, p->complex_cost, p->dcomplex_cost);
	}
	else
	    skip_one_line = TRUE;
    }
    fprintf(fd, "\n");
}

/* Completes the intrinsic cost table with the costs read from the files
 * specified in the "COMPLEXITY_COST_TABLE" string property
 * See properties.rc and ~pips/Pips/pipsrc.csh for more information.
 * 
 * L. ZHOU 13/03/91
 *
 * COST_DATA are names of five data files
 */
void init_cost_table()
{
    char *sep_chars = strdup(" ");
    char *token, *comma, *file = (char*) malloc(80);
    float file_factor;

    char *cost_dir = getenv("PIPS_COSTDIR");
    char *cost_table = strdup(get_string_property("COMPLEXITY_COST_TABLE"));
    char *cost_data = strdup(COST_DATA);
    char *tmp= (char*) malloc(20);

    if (!cost_dir) /* the default value */
	cost_dir = strdup(concatenate(getenv("PIPS_ROOT"),
				      "/Share/complexity_cost_tables", NULL));
    else
	cost_dir = strdup(cost_dir);

    pips_assert("some directory and table", cost_dir && cost_table);
    token = strtok(cost_data, sep_chars);

    while (token != NULL) {
	comma = strchr(token, ',');

	if (comma == NULL) {
	    strcpy(tmp, token);
	    file_factor = 1.0;
	}
	else {
	    int ii = comma - token;
	    strncpy(tmp, token, ii);
	    *(tmp + ii) = '\0';
	    sscanf(++comma, "%f", &file_factor);
	}


	file = concatenate(cost_dir,"/", cost_table, "/", tmp, NULL);

	debug(5,"init_cost_table","file_factor is %f\n", file_factor);
	debug(1,"init_cost_table","cost file is %s\n",file);
	load_cost_file(file, file_factor);
	token = strtok(NULL, sep_chars);
    }

    free(cost_dir);
}

/* 
 * Load (some) intrinsics costs from file "filename", 
 * multiplying them by "file_factor".
 */
void load_cost_file(filename, file_factor)
char *filename;
float file_factor;
{
    FILE *fd;
    char *line = (char*) malloc(199);
    char *intrinsic_name = (char*) malloc(30);
    int int_cost, float_cost, double_cost, complex_cost, dcomplex_cost;
    struct intrinsic_cost_rec *p;
    float scale_factor = 1.0;
    boolean recognized;

    if (!file_exists_p(filename))
	user_error("load_cost_table",
		   "This cost file: %s doesn't exist\n",
		   filename);
    else {
	fd = safe_fopen(filename, "r");
	
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "\nReading cost file '%s' ", filename);
	    if (file_factor != 1.0) 
		fprintf(stderr, "(x %.2f)", file_factor);
	}
	
	while (fgets(line, 99, fd) != NULL) {
	    if (*line == '%')
		sscanf(line+1, "%f", &scale_factor);
	    else if ((*line != '#') && (*line != '\n')) {
		sscanf(line, "%s %d %d %d %d %d", intrinsic_name,
		       &int_cost, &float_cost, &double_cost,
		       &complex_cost, &dcomplex_cost);
		recognized = FALSE;
		for (p = intrinsic_cost_table; p->name != NULL; p++) {
		    if (streq(p->name, intrinsic_name)) {
			p->int_cost = (int)
			    (int_cost * scale_factor * file_factor + 0.5);
			p->float_cost = (int)
			    (float_cost * scale_factor * file_factor + 0.5);
			p->double_cost = (int)
			    (double_cost * scale_factor * file_factor + 0.5);
			p->complex_cost = (int)
			    (complex_cost * scale_factor * file_factor + 0.5);
			p->dcomplex_cost = (int)
			    (dcomplex_cost * scale_factor * file_factor + 0.5);
			recognized = TRUE;
			break;
		    }
		}
		if (!recognized)
		    user_warning("load_cost_file",
				 "%s:unrecognized intrinsic\n",intrinsic_name);
	    }
	}
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "\nScale factor is %f\n", scale_factor);
	}
	safe_fclose(fd, filename);
    }
 
    free(intrinsic_name);
    free(line);
}

/* Return the cost of the intrinsic named s, knowing that
 * the "basic" type of its biggest argument is *pargsbasic.
 * Update *pargsbasic if the intrinsic returns a number
 * of bigger complexity.
 */
int intrinsic_cost(s, pargsbasic)
char *s;
basic *pargsbasic;
{
  struct intrinsic_cost_rec *p;
  basic b;

  for (p = intrinsic_cost_table; p->name != NULL; p++) {
    if (streq(p->name, s)) {

      /* Inserted by AP, oct 24th 1995 */
      if (streq(p->name, "LOG") || streq(p->name, "LOG10")) {
	user_warning("intrinsic_cost", "LOG or LOG10 functions used\n");
      }

      b = make_basic(p->min_basic_result, p->min_nbytes_result);
      if (is_inferior_basic(*pargsbasic, b)) {
	free_basic(*pargsbasic);
	*pargsbasic = simple_basic_dup(b);
      }

      switch (basic_tag(*pargsbasic)) {
	case is_basic_int:
	  return(p->int_cost);
	case is_basic_float:
	  return (basic_float(*pargsbasic) <= FLOAT_NBYTES ?
		  p->float_cost : p->double_cost);
	case is_basic_complex:
	  return (basic_complex(*pargsbasic) <= COMPLEX_NBYTES ?
		  p->complex_cost : p->dcomplex_cost);
	case is_basic_string:
	  return (STRING_INTRINSICS_COST);
	case is_basic_logical:
	  return (LOGICAL_INTRINSICS_COST);
	default:
	  pips_error("intrinsic_cost",
		     "basic tag is %d\n", basic_tag(*pargsbasic));
	}
    }
  }
  /* To satisfy cproto . LZ 02 Feb. 93 */
  return (STRING_INTRINSICS_COST); 
}


/* Return if possible the value of e in a float.
 * it is supposed to be an int or a float.
 */
float constant_entity_to_float(e)
entity e;
{
    char *cste = module_local_name(e);
    basic b = entity_basic(e);
    float f;

    if (basic_int_p(b) || basic_float_p(b)) {
	sscanf(cste, "%f", &f);
	return (f);
    }
    else {
	user_warning("constant_entity_to_float",
		     "Basic tag:%d, not 4->9, (entity %s)\n",basic_tag(b),cste);
	return (0.0);
    }
}

/* "trace on" */
static int call_level=0;
void trace_on(char * fmt, ...)
{
    if (get_bool_property("COMPLEXITY_TRACE_CALLS")) {
	va_list args;
	char *indentstring = (char*) malloc(99);
	boolean b = (call_level >= 0);
	int i,k=1;

	indentstring[0] = '\0';

	for (i=0; i< (b ? call_level : - call_level); i++) {
	    indentstring = strcat(indentstring,
				  strdup(b ? ( (k>0) ? INDENT_BLANKS
					             : INDENT_VLINE )
					   : INDENT_BACK));
	    k = ( k<INDENT_INTERVAL ? k+1 : 0 );
	}

	fprintf(stderr, "%s>", indentstring);
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);

	free(indentstring);
	call_level++;
    }
}

/* "trace off" */
void trace_off()
{
    if (get_bool_property("COMPLEXITY_TRACE_CALLS")) {
	char *indentstring = (char*) malloc(99);
	boolean b = (call_level >= 0);
	int i,k=1;

	indentstring[0] = '\0';
	call_level--;
	for (i=0; i< (b ? call_level : - call_level); i++) {
	    indentstring = strcat(indentstring,
				  strdup(b ? ( (k>0) ? INDENT_BLANKS
					             : INDENT_VLINE )
					   : INDENT_BACK));
	    k = ( k<INDENT_INTERVAL ? k+1 : 0 );
	}
	fprintf(stderr, "%s<\n", indentstring);
	free(indentstring);
    }
}

/* return TRUE if unstr is simply a linear 
 * string of controls
 */
boolean is_linear_unstructured(unstr)
unstructured unstr;
{
    control current = unstructured_control(unstr);
    control exit = unstructured_exit(unstr);

    while (current != exit) {
	list succs = control_successors(current);

	if (succs == NIL)
	    pips_error("is_linear_unstructured",
		       "control != exit one,it has no successor\n");
	if (CDR(succs) != NIL) 
	    return (FALSE);
	current = CONTROL(CAR(succs));
    }

    return(TRUE);
}

list entity_list_reverse(l)
list l;
{
    entity e;

    if ((l == NIL) || (l->cdr == NIL)) 
	return l;
    e = ENTITY(CAR(l));
    return (CONS(ENTITY, e, entity_list_reverse(l->cdr)));
}

void add_formal_parameters_to_hash_table(mod, hash_complexity_params)
entity mod;
hash_table hash_complexity_params;
{
    list decl;

    pips_assert("add_formal_parameters_to_hash_table",
		entity_module_p(mod));
    decl = code_declarations(value_code(entity_initial(mod)));

    MAPL(pe, {
	entity param = ENTITY(CAR(pe));
	if (storage_formal_p(entity_storage(param))) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr,"\nstorage_formal %s\n",
			entity_name(param));
	    }
	    hash_put(hash_complexity_params, (char *) module_local_name(param),
		     HASH_FORMAL_PARAM);
        }
    }, decl);
}

void remove_formal_parameters_from_hash_table(mod, hash_complexity_params)
entity mod;
hash_table hash_complexity_params;
{
    list decl;

    pips_assert("remove_formal_parameters_from_hash_table",
		entity_module_p(mod));
    decl = code_declarations(value_code(entity_initial(mod)));

    MAPL(pe, {
	entity param = ENTITY(CAR(pe));
	if (storage_formal_p(entity_storage(param)))
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr,"storage_formal %s to be deleted\n",
			entity_name(param));
	    }
	    hash_del(hash_complexity_params, (char *) module_local_name(param));
    }, decl);
}

hash_table free_callees_complexities(hash_table h)
{
    /* Modified copies of the summary complexities are stored */
    hash_table_clear(h);
    hash_table_free(h);

    return hash_table_undefined;
}

hash_table fetch_callees_complexities(module_name)
char *module_name;
{
    hash_table hash_callees_comp = hash_table_make(hash_pointer, 0);
    callees cl;
    list callees_list;
    complexity callee_comp;

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching callees complexities ...\n");
    }

    cl = (callees)db_get_memory_resource(DBR_CALLEES, module_name, TRUE);
    callees_list = callees_callees(cl);

    if ( callees_list == NIL ) { 
	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "Module %s has no callee! Done\n", module_name);
    	}
	return(hash_callees_comp);
    }
 
    MAPL(pc, {
	string callee_name = STRING(CAR(pc));
	entity callee = local_name_to_top_level_entity(callee_name);
	type t = entity_type(callee);

	if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	    fprintf(stderr, "%s has callee %s!\n",module_name,callee_name);
	}
	pips_assert("call_to_complexity",
		    type_functional_p(t) || type_void_p(t));

	if (value_code_p(entity_initial(callee))) {
	    complexity new_comp;
	    callee_comp = (complexity)
		db_get_memory_resource(DBR_SUMMARY_COMPLEXITY, 
				       (char *) callee_name,
				       TRUE);

	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "fetched complexity for callee %s",
			         callee_name);
		fprintf(stderr, " of module %s:\n", module_name);
		complexity_fprint(stderr, callee_comp, 
				          DO_PRINT_STATS, 
				          ! PRINT_LOCAL_NAMES);
	    }

	    debug(5,"fetch_callees_complexities","callee_name %s\n",callee_name);

	    /* translate the local name to current module name. LZ 5 Feb.93 */
	    /* i.e. SUB:M -> MAIN:M */
	    /* FI: this seems to be wrong in general because the 
	     * formal parameter and actual argument are assumed to
	     * have the same name; see DemoStd/q and variables IM/IMM;
	     * 3 March 1994
	     */
	    new_comp = translate_complexity_from_local_to_current_name(callee_comp, 
							    callee_name,module_name);

	    hash_put(hash_callees_comp, (char *)callee, (char *)new_comp);
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "translated complexity for callee %s",
			         callee_name);
		fprintf(stderr, " of module %s:\n", module_name);
		complexity_fprint(stderr, new_comp, 
				          DO_PRINT_STATS, 
				          ! PRINT_LOCAL_NAMES);
	    }
	}
    }, callees_list );

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching callees complexities ... done\n");
    }

    return(hash_callees_comp);
}

hash_table fetch_complexity_parameters(module_name)
char *module_name;
{
    hash_table hash_comp_params = hash_table_make(hash_pointer, 0);
    char *parameters = strdup(get_string_property("COMPLEXITY_PARAMETERS"));
    char *sep_chars = strdup(", ");
    char *token = (char*) malloc(30);
    entity e;

    hash_warn_on_redefinition();
   
    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching complexity parameters for module %s:\n",
		module_name);
    }

    token = strtok(parameters, sep_chars);

    while (token != NULL) {
	e = gen_find_tabulated(concatenate(module_name,
					   MODULE_SEP_STRING,
					   token,
					   (char *) NULL),
			       entity_domain);
	if (e != entity_undefined) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "{\t Defined entity  %s }\n", entity_name(e));
	    }
	    hash_put(hash_comp_params,(char *)e,HASH_USER_VARIABLE);
	}
	else {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "{\t Undefined token  %s }\n", token);
	    }
	}
	token = strtok(NULL, sep_chars);
    }

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "Fetching complexity parameters: ...done.\n");
    }

    return(hash_comp_params);
}

void add_common_variables_to_hash_table(module, hash_complexity_params)
entity module;
hash_table hash_complexity_params;
{
    string module_name = module_local_name(module);
    list sefs_list = list_undefined;

    pips_assert("add_common_variables_to_hash_table",
		entity_module_p(module));

    sefs_list = effects_to_list( (effects)
	db_get_memory_resource(DBR_SUMMARY_EFFECTS, module_name, TRUE));

    ifdebug(5) {
	debug(5, "add_common_variables_to_hash_table",
	      "Effect list for %s\n",
	      module_name);
	print_effects(sefs_list);
    }

    MAPL(ce, { 
	effect obj = EFFECT(CAR(ce));
	reference r = effect_reference(obj);
	action ac = effect_action(obj);
	approximation ap = effect_approximation(obj);
	entity e = reference_variable(r);
	storage s = entity_storage(e);

	if ( !storage_formal_p(s) &&
	    action_read_p(ac) && approximation_must_p(ap) ) {
	    debug(5,"add_common_variables_to_hash_table",
		  "%s added\n", module_local_name(e));
	    hash_put(hash_complexity_params, (char *) module_local_name(e),
		     HASH_COMMON_VARIABLE);
	}
    }, sefs_list);
}

void remove_common_variables_from_hash_table(module, hash_complexity_params)
entity module;
hash_table hash_complexity_params;
{
    string module_name = module_local_name(module);
    list sefs_list;

    pips_assert("remove_common_variables_from_hash_table",
		entity_module_p(module));

    sefs_list = effects_to_list( (effects)
	db_get_memory_resource(DBR_SUMMARY_EFFECTS, module_name, TRUE));

    MAPL(ce, { 
	effect obj = EFFECT(CAR(ce));
	reference r = effect_reference(obj);
	action ac = effect_action(obj);
	approximation ap = effect_approximation(obj);
	entity e = reference_variable(r);

	if ( action_read_p(ac) && approximation_must_p(ap) ) {
	    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
		fprintf(stderr, "%s deleted\n", module_local_name(e));
	    }
	    hash_del(hash_complexity_params, (char *) module_local_name(e));
	}
    }, sefs_list);
}

boolean is_must_be_written_var(effects_list, var_name)
list effects_list;
char *var_name;
{
    MAPL(ce, { 
	effect eff = EFFECT(CAR(ce));

	if(eff == effect_undefined)
	    pips_error("is_must_be_written_var", "unexpected effect undefined");

	if ( action_write_p(effect_action(eff)) 
	    && approximation_must_p(effect_approximation(eff)) ) {
	    reference r = effect_reference(eff);
	    entity e = reference_variable(r);
/*	    
	    fprintf(stderr, "is_must_be_written_var for entity %s\n", 
		    module_local_name(e) );
*/
	    if ( strcmp(module_local_name(e), var_name) == 0 ) {
		return (TRUE);
	    }
	}
/*
	else {
	    fprintf(stderr, "is_must_be_written_var for NOT entity %s\n", 
		    module_local_name(reference_variable(effect_reference(eff))) );
	}
*/
    },effects_list);

    return (FALSE);
}

/*
 * This procedure is used to evaluate the complexity which has been postponed 
 * to be evaluated by is_must_be_writteen.
 * LZ 26 Nov. 92
 */
complexity final_statement_to_complexity_evaluation(comp, precond, effects_list)
complexity comp;
transformer precond;
list effects_list;
{
    complexity final_comp = complexity_dup(comp);
    Ppolynome pp = complexity_polynome(comp);
    extern boolean default_is_inferior_pvarval(Pvecteur *, Pvecteur *);
    Pbase pb = vect_dup(polynome_used_var(pp, default_is_inferior_pvarval));


    fprintf(stderr, "Final evaluation\n");

    for ( ; !VECTEUR_NUL_P(pb); pb = pb->succ) {
	boolean mustbewritten;
	char *var = variable_local_name(pb->var);

        fprintf(stderr, "Variable is %s\n", var);

        mustbewritten = is_must_be_written_var(effects_list, var);

        if ( mustbewritten ) {
	    complexity compsubst;
	    fprintf(stderr, "YES once\n");
	    compsubst = evaluate_var_to_complexity((entity)pb->var, 
						   precond, 
						   effects_list, 1);
	    complexity_fprint( stderr, compsubst, FALSE, FALSE);
/*

	    final_comp = complexity_var_subst(comp, pb->var, compsubst);
*/
	}
	comp = complexity_dup(final_comp);
    }

    complexity_fprint( stderr, final_comp, FALSE, FALSE);

    return ( final_comp );
}

/* translate_complexity_from_local_to_current_name(callee_comp,oldname,newname)
 * B:M -> A:M if A calls B
 * 5 Feb. 93 LZ
 *
 * This is not general enough to handle:
 * B:M -> A:N or B:M to A:N+1
 * FI, 3 March 1994
 */
complexity translate_complexity_from_local_to_current_name(callee_comp,oldname,newname)
complexity callee_comp;
string oldname,newname;
{
    Ppolynome pp = complexity_polynome(callee_comp);
    extern boolean is_inferior_pvarval(Pvecteur *, Pvecteur *);
    Pbase pb = polynome_used_var(pp, is_inferior_pvarval);
    Pbase pbcur = BASE_UNDEFINED;
    complexity comp = make_zero_complexity();
    complexity old_comp = complexity_dup(callee_comp);

    if(BASE_NULLE_P(pb)) {
	/* constant complexity */
	comp = complexity_dup(callee_comp);
	return comp;
    }

    /* The basis associated to a polynomial includes the constant term! */
    if(base_dimension(pb)==1 && term_cst(pb)) {
	/* constant complexity */
	comp = complexity_dup(callee_comp);
	return comp;
    }

    for (pbcur=pb; pbcur != VECTEUR_NUL ; pbcur = pbcur->succ ) {
	Variable var = pbcur->var;
	char * stmp = strdup(variable_name(var));

	char *s = stmp;
	char *t = strchr(stmp,':');

	if ( t != NULL ) {
	    int length = (int)(t - s);
	    char *cur_name = (char *)malloc(100);

	    (void) strncpy(cur_name,stmp,length);
	    * (cur_name+length) = '\0';

	    if ( 1 || strncmp(cur_name, oldname, length) == 0 ) {
		Variable newvar = name_to_variable(concatenate(strdup(newname),
							       ":",strdup(t+1),NULL));
		if ( newvar != (Variable) chunk_undefined ) {
		    complexity compsubst = make_single_var_complexity(1.0, newvar);

/*
                    polynome_chg_var(&pp, var, newvar);
*/
		    comp = complexity_var_subst(old_comp, var, compsubst);
		    old_comp = complexity_dup(comp);
		}
		else {
		    comp = complexity_dup(old_comp);
		    old_comp = complexity_dup(comp);
		}		    
	    }
	}
    }
    return (comp);
}

bool complexity_is_monomial_p(complexity c)
{
    Ppolynome p = complexity_eval(c);
    bool monomial_p = is_single_monome(p);

    return monomial_p;
}

int complexity_degree(complexity c)
{
    Ppolynome p = complexity_eval(c);
    int degree = polynome_max_degree(p);

    return degree;
}
