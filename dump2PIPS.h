/*
* includes for newgen
* a link to the newgen binary file is missing
*/

#undef toupper
#ifndef GFC_2_PIPS
#define GFC_2_PIPS

//hack : conflict for list which is defined as a type in PIPS and as a variable in gfc
#undef list
#define list newgen_list

//hack : bool/boolean defined in both gfc and PIPS and in different ways
#define boolean unsigned char
#undef bool
#define BOOLEAN_INCLUDED

//hack : match
#define match pips_match

//hack : After gfortran.h free become a macro for another function
#undef free

#include "genC.h"

#define NIL ((newgen_list)(void*)0)


#include "linear.h"
#include "parser_private.h"
#include "ri.h"

#include "makefile.h"
#include "pipsmake.h"
#include "control.h"
#include "text.h"
#include "prettyprint.h"

//#include "/data/raphael/pips/prod/pips/src/Libs/syntax/procedure.c"
#include "database.h"
#include "resources.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "syntax.h"

#include "/data/raphael/pips/prod/pips/src/Libs/syntax/syn_yacc.h"
//#include "/data/raphael/pips/prod/pips/src/Libs/newgen/newgen.c"

#undef match
#undef list





extern type CurrentType;
newgen_list gfc_called_modules;
statement gfc_function_body;


char * str2upper(char s[]);
char * strn2upper(char s[], size_t n);

/*
 * Dump a namespace
 */
void gfc2pips_namespace(gfc_namespace* ns);

/*
 * Return a list of every and each arguments for PIPS from a  gfc function/subroutine
 */
newgen_list gfc2pips_args(gfc_namespace* ns);

/*
 * Find a symbol by it reference name
 * Simplified version of
 * static gfc_symtree * find_symbol (gfc_symtree *st, const char *name, const char *module, int generic);
 */
gfc_symtree* getSymtreeByName (char* name, gfc_symtree *st);

newgen_list gfc2pips_vars(gfc_namespace *ns);
newgen_list gfc2pips_vars_(gfc_namespace *ns,newgen_list variables_p);

/*
 * Find a list of symbols if they verify the predicate function
 */
newgen_list getSymbolBy(gfc_namespace* ns, gfc_symtree *st, bool (*func)(gfc_namespace*, gfc_symtree *));

/*
 * Predicate functions
 */
bool gfc2pips_test_variable(gfc_namespace* ns, gfc_symtree *st);
bool gfc2pips_test_variable2(gfc_namespace* ns, gfc_symtree *st );
//bool gfc2pips_test_name(gfc_namespace* ns, gfc_symtree *st, int param);
bool gfc2pips_test_data(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );

entity gfc2pips_symbol2entity(gfc_symbol* sym);

/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */
dimension gfc2pips_int2dimension(int n);

expression gfc2pips_int2expression(int n);//PIPS: expression int_to_expression(_int)
expression gfc2pips_real2expression(double r);
expression gfc2pips_logical2expression(bool b);
expression gfc2pips_string2expression(char* s);

entity gfc2pips_int_const2entity(int n);
entity gfc2pips_int2label(int n);
entity gfc2pips_real2entity(double r);
entity gfc2pips_logical2entity(bool b);
char* gfc2pips_gfc_char_t2string(gfc_char_t *c,int nb);
char* gfc2pips_gfc_char_t2string_(gfc_char_t *c,int nb);

value gfc2pips_symbol2value(gfc_symbol *s);

type gfc2pips_symbol2type(gfc_symbol *s);
int gfc2pips_symbol2size(gfc_symbol *s);

instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c);
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence);
instruction gfc2pips_code2instruction_(gfc_code* c);
instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym);
entity gfc2pips_code2get_label(gfc_code *c);
entity gfc2pips_code2get_label2(gfc_code *c);

expression gfc2pips_expr2expression(gfc_expr *expr);
bool gfc2pips_exprIsVariable(gfc_expr * expr);
entity gfc2pips_expr2entity(gfc_expr *expr);



#endif /* GFC_2_PIPS */

