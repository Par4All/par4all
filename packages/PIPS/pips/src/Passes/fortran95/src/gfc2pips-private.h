/*

 $Id$

 Copyright 1989-2014 MINES ParisTech
 Copyright 2009-2010 HPC-Project

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

/*
* includes for newgen
* a link to the newgen binary file is missing
*/

#ifndef GFC_2_PIPS_PRIVATE
#define GFC_2_PIPS_PRIVATE


#include "genC.h"
#include "gfc2pips.h"

#include "linear.h"
#include "ri.h"
#include "ri-util.h"


#include "parser_private.h"
#include "syntax.h"


/* We have here to HACK GCC include that prevent use of these function */
#undef toupper
#undef fgetc
#undef fputc
#undef fread
#undef asprintf
int asprintf(char **strp, const char *fmt, ...);





#define gfc2pips_debug pips_debug

//an enum to know what kind of main entity we are dealing with
typedef enum gfc2pips_main_entity_type {
  MET_PROG, MET_SUB, MET_FUNC, MET_MOD, MET_BLOCK, MET_MODULE
} gfc2pips_main_entity_type;


/* Store the list of callees */
extern list gfc_module_callees;


extern list gfc2pips_list_of_declared_code;
extern list gfc2pips_list_of_loops;


void gfc2pips_add_to_callees(entity e);
void pips_init();
list get_use_entities_list(struct gfc_namespace *ns);
void save_entities();
basic gfc2pips_getbasic(gfc_symbol *s);

/**
 * @brief put the given char table to upper case
 */
char * str2upper(char s[]);
/**
 * @brief put the n first elements of the given char table to upper case
 */
char * strn2upper(char s[], size_t n);
/**
 * @brief same as strcpy, but begin by the end of the string allowing you to give twice the same string
 */
char * strrcpy(char *dest, __const char *src);
/**
 * @brief compare the strings in upper case mode
 */
int strcmp_ (__const char *__s1, __const char *__s2);
/**
 * @brief compare the strings in upper case mode
 */
int strncmp_ (__const char *__s1, __const char *__s2, size_t __n);
/**
 * @brief copy the content of the first file to the second one
 */
int fcopy(const char* old, const char* new );


void gfc2pips_truncate_useless_zeroes(char *s);



/*
 * Dump a namespace
 */
void gfc2pips_namespace(gfc_namespace* ns);

/*
 * Return a list of every and each arguments for PIPS from a  gfc function/subroutine
 */
list gfc2pips_args(gfc_namespace* ns);

void gfc2pips_generate_parameters_list(list parameters);

/*
 * Find a symbol by it reference name
 * Simplified version of
 * static gfc_symtree * find_symbol (gfc_symtree *st, const char *name, const char *module, int generic);
 */
gfc_symtree* gfc2pips_getSymtreeByName (const char* name, gfc_symtree *st);

gfc2pips_main_entity_type get_symbol_token( gfc_symbol *root_sym );

list gfc2pips_vars(gfc_namespace *ns);
list gfc2pips_parameters( gfc_namespace * ns,
                          gfc2pips_main_entity_type bloc_token );
list gfc2pips_vars_(gfc_namespace *ns,list variables_p);
void gfc2pips_getTypesDeclared(gfc_namespace *ns);
list gfc2pips_get_extern_entities(gfc_namespace *ns);
list gfc2pips_get_data_vars(gfc_namespace *ns);
list gfc2pips_get_save(gfc_namespace *ns);

list gfc2pips_get_list_of_dimensions(gfc_symtree *st);
list gfc2pips_get_list_of_dimensions2(gfc_symbol *s);

/*
 * Find a list of symbols if they verify the predicate function
 */
list getSymbolBy(gfc_namespace* ns, gfc_symtree *st, bool (*func)(gfc_namespace*, gfc_symtree *));

/*
 * Predicate functions
 */
bool gfc2pips_test_variable(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st);
bool gfc2pips_test_variable2(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );
bool gfc2pips_test_derived(gfc_namespace __attribute__ ((__unused__)) *ns, gfc_symtree *st );
bool gfc2pips_test_extern(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );
bool gfc2pips_test_subroutine(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );
//bool gfc2pips_test_name(gfc_namespace* ns, gfc_symtree *st, int param);
bool gfc2pips_test_data(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );
bool gfc2pips_test_save(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree *st );
bool gfc2pips_get_commons(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* __attribute__ ((__unused__)) st );
bool gfc2pips_get_incommon(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* __attribute__ ((__unused__)) st );
bool gfc2pips_test_dimensions(gfc_namespace* __attribute__ ((__unused__)) ns, gfc_symtree* st );

entity gfc2pips_check_entity_doesnt_exists(char *s);
entity gfc2pips_check_entity_program_exists(char *s);
entity gfc2pips_check_entity_module_exists(char *s);
entity gfc2pips_check_entity_block_data_exists(char *s);
entity gfc2pips_check_entity_exists(const char *s);
entity gfc2pips_symbol2entity(gfc_symbol* sym);
entity gfc2pips_symbol2top_entity(gfc_symbol* sym);
entity gfc2pips_char2entity(char*p, char* s);
char* gfc2pips_get_safe_name(const char* str);


/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */
dimension gfc2pips_int2dimension(int n);

expression gfc2pips_int2expression(int n);//PIPS: expression int_to_expression(_int)
expression gfc2pips_real2expression(double r);
expression gfc2pips_logical2expression(bool b);

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

list gfc2pips_array_ref2indices(gfc_array_ref *ar);
bool gfc2pips_there_is_a_range(gfc_array_ref *ar);
list gfc2pips_mkRangeExpression(gfc_array_ref *ar);


instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c);
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence);
instruction gfc2pips_code2instruction_(gfc_code* c);
expression gfc2pips_buildCaseTest(gfc_expr *tested_variable, gfc_case *cp);
list gfc2pips_dumpSELECT(gfc_code *c);
instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym);
expression gfc2pips_make_zero_for_symbol(gfc_symbol* sym);
list gfc2pips_reduce_repeated_values(list l);
entity gfc2pips_code2get_label(gfc_code *c);
entity gfc2pips_code2get_label2(gfc_code *c);
entity gfc2pips_code2get_label3(gfc_code *c);
entity gfc2pips_code2get_label4(gfc_code *c);

expression gfc2pips_expr2expression(gfc_expr *expr);
int gfc2pips_expr2int(gfc_expr *expr);
bool gfc2pips_exprIsVariable(gfc_expr * expr);
entity gfc2pips_expr2entity(gfc_expr *expr);

//translate an expression or a value of a IO statement
list gfc2pips_exprIO(char* s, gfc_expr* e, list l);
list gfc2pips_exprIO2(char* s, int e, list l);
list gfc2pips_exprIO3(char* s, string e, list l);

list gfc2pips_arglist2arglist(gfc_actual_arglist *act);

//memory related functions
void gfc2pips_initAreas(void);
void gfc2pips_computeAdresses(void);
void gfc2pips_computeAdressesStatic(void);
void gfc2pips_computeAdressesDynamic(void);
void gfc2pips_computeAdressesHeap(void);
int gfc2pips_computeAdressesOfArea( entity _area );
void gfc2pips_shiftAdressesOfArea( entity _area, int old_offset, int size, int max_offset, int shift );

list *gfc2pips_list_of_all_modules;


void gfc2pips_push_comment(locus l, unsigned long nb, char s);
bool gfc2pips_check_already_done(locus l);
unsigned long gfc2pips_get_num_of_gfc_code(gfc_code *c);
string gfc2pips_get_comment_of_code(gfc_code *c);
gfc2pips_comments gfc2pips_pop_comment(void);
//void gfc2pips_set_last_comments_done(gfc_code *c);
void gfc2pips_set_last_comments_done(unsigned long nb);
void gfc2pips_assign_num_to_last_comments(unsigned long nb);
void gfc2pips_assign_gfc_code_to_last_comments(gfc_code *c);
void gfc2pips_replace_comments_num(unsigned long old, unsigned long new);
void gfc2pips_assign_gfc_code_to_num_comments(gfc_code *c, unsigned long num);
bool gfc2pips_comment_num_exists(unsigned long num);
void gfc2pips_pop_not_done_comments(void);

void gfc2pips_shift_comments(void);

void gfc2pips_push_last_code(gfc_code *c);

gfc_code* gfc2pips_get_last_loop(void);
void gfc2pips_push_loop(gfc_code *c);
void gfc2pips_pop_loop(void);

list gen_union(list a, list b);

#endif /* GFC_2_PIPS */

