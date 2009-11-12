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
#define hash_table pips_hash_table
#define loop pips_loop

//hack : After gfortran.h free become a macro for another function
#undef free

#include <stdio.h>
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

#undef loop
#undef hash_table
#undef match
#undef list

extern int gfc2pips_nb_of_statements;


typedef struct _gfc2pips_comments_{
	bool done;
	locus l;
	gfc_code *gfc;
	char *s;
	unsigned long num;
	struct _gfc2pips_comments_ *prev;
	struct _gfc2pips_comments_ *next;
} *gfc2pips_comments;


extern gfc2pips_comments gfc2pips_comments_stack;
extern gfc2pips_comments gfc2pips_comments_stack_;

extern type CurrentType;
newgen_list gfc_called_modules;
statement gfc_function_body;






char * str2upper(char s[]);
char * strn2upper(char s[], size_t n);
char * strrcpy(char *dest, __const char *src);
int fcopy(char* ,char* );

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

newgen_list gfc2pips_get_list_of_dimensions(gfc_symtree *st);
newgen_list gfc2pips_get_list_of_dimensions2(gfc_symbol *s);

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
bool gfc2pips_get_commons(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* __attribute__ ((__unused__)) st );
bool gfc2pips_test_dimensions(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* st );

entity gfc2pips_symbol2entity(gfc_symbol* sym);
entity gfc2pips_char2entity(char*p, char* s);

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
char* gfc2pips_gfc_char_t2string2(gfc_char_t *c);
char* gfc2pips_gfc_char_t2string_(gfc_char_t *c,int nb);

value gfc2pips_symbol2value(gfc_symbol *s);

type gfc2pips_symbol2type(gfc_symbol *s);
int gfc2pips_symbol2size(gfc_symbol *s);
int gfc2pips_symbol2sizeArray(gfc_symbol *s);

instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c);
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence);
instruction gfc2pips_code2instruction_(gfc_code* c);
instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym);
entity gfc2pips_code2get_label(gfc_code *c);
entity gfc2pips_code2get_label2(gfc_code *c);
entity gfc2pips_code2get_label3(gfc_code *c);
entity gfc2pips_code2get_label4(gfc_code *c);

expression gfc2pips_expr2expression(gfc_expr *expr);
int gfc2pips_expr2int(gfc_expr *expr);
bool gfc2pips_exprIsVariable(gfc_expr * expr);
entity gfc2pips_expr2entity(gfc_expr *expr);

//translate an expression or a value of a IO statement
newgen_list gfc2pips_exprIO(char* s, gfc_expr* e, newgen_list l);
newgen_list gfc2pips_exprIO2(char* s, int e, newgen_list l);

void gfc2pips_initAreas();

newgen_list *gfc2pips_list_of_all_modules;


void gfc2pips_push_comment(locus l, unsigned long nb, char s);
bool gfc2pips_check_already_done(locus l);
unsigned long gfc2pips_get_num_of_gfc_code(gfc_code *c);
string gfc2pips_get_comment_of_code(gfc_code *c);
gfc2pips_comments gfc2pips_pop_comment();
//void gfc2pips_set_last_comments_done(gfc_code *c);
void gfc2pips_set_last_comments_done(unsigned long nb);
void gfc2pips_assign_num_to_last_comments(unsigned long nb);
void gfc2pips_assign_gfc_code_to_last_comments(gfc_code *c);
void gfc2pips_replace_comments_num(unsigned long old, unsigned long new);
void gfc2pips_assign_gfc_code_to_num_comments(gfc_code *c, unsigned long num);
bool gfc2pips_comment_num_exists(unsigned long num);
void gfc2pips_pop_not_done_comments();

void gfc2pips_shift_comments();

void gfc2pips_push_last_code(gfc_code *c);


#endif /* GFC_2_PIPS */

