/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


/* include.h */

#include "newgen_types.h"
#include "newgen_set.h"

/* Types of domain */

#define UNDEF 0
#define EXTERNAL 1
#define BASIS 2
#define LIST 3
#define ARRAY 4
#define CONSTRUCTED 5
#define IMPORT 6
#define SET 7

/* Operators for CONSTRUCTED domains */

#define UNDEF_OP 0
#define AND_OP 1
#define OR_OP 2
#define ARROW_OP 3

/* Names of CONSTRUCTED operators */

extern char *Op_names[] ;

/* The UNIT_TYPE is the used to type expressions which only perform 
   side-effects. It is mainly used to build sets. */

#define UNIT_TYPE "unit"

/* A list type constructor */

#define DEFLIST(type,cartype,carname)\
  struct type {\
    cartype carname ;\
    struct type *cdr ;\
    }

DEFLIST(namelist,char *,name) ;
DEFLIST(intlist,int,val) ;
DEFLIST(domainlist,union domain *,domain) ;

struct gen_binding ;

/* A DOMAIN union describes the structure of a user type. 

   The (STRUCT BINDING *) are used either to point to strings (during
   the parsing of the specifications), or to gen_bindings (after compilation).

   TYPE and CONSTRUCTOR members have to be at the same offset */

/* FC, 10/06/94, set_type in se moved.
 *
 * common part assumed for all: type
 * common part assumed for ba, li, se, ar: 
 *      type, constructor, persistant and one gen_binding
 *
 */
union domain {
    struct {
	int type ;
	char *(*read)() ;
	void (*write)() ;
	void (*free)() ;
	char *(*copy)() ;
    } ex ;
    struct { 
	int type ;
	char *constructor ;
	int persistant ;
	struct gen_binding *constructand ;
    } ba ;
    struct {
	int type ;
	char *constructor ;
	int persistant ;
	struct gen_binding *element ;
    } li ;
    struct {
	int type ;
	char *constructor ;
	int persistant ;
	struct gen_binding *element ;
	set_type what ;
    } se ;
    struct {
	int type ;
	char *constructor ;
	int persistant ;
	struct gen_binding *element ;
	struct intlist *dimensions ;
    } ar ;
    struct {
	int type ;
	int op ;
	int first ;
	struct domainlist *components ;
    } co ;
    struct {
	int type ;
	char *filename ;
    } im ;
} ;

/* MAX_DOMAIN is the maximum number of entries in the DOMAINS table */

#define MAX_DOMAIN 250

/* For tabulated types. */

#define MAX_TABULATED 10

/* FI: huge increas to cope with Renault's code :-(
 * #define MAX_TABULATED_ELEMENTS 12013
 */
#define MAX_TABULATED_ELEMENTS 200003

extern struct gen_binding *Tabulated_bp ;

/* INLINE[] gives, for each inlinable (i.e., unboxed) type, its NAME,
   its initial VALUE and its printing FORMAT (for each language which can
   be a target. */

extern struct inlinable {
  char *name ;
  char *C_value ;
  char *C_format ;
  char *Lisp_value ;
  char *Lisp_format ;
} Inline[] ;

/* Different kinds of BINDING structure pointers */

#define IS_INLINABLE(bp) ((bp)->inlined != NULL)
#define IS_EXTERNAL(bp) ((bp)->domain->ba.type == EXTERNAL)
#define IS_TABULATED(bp) ((bp)->index >= 0)
#define IS_IMPORT(bp) ((bp)->domain->ba.type == IMPORT)
#define IS_NORMAL(bp) \
	(!IS_INLINABLE(bp)&&!IS_EXTERNAL(bp)&&!IS_TABULATED(bp))

/* Domains is the symbol table for user defined types. COMPILED refers to
   inlined data types. INDEX is used whenever the type is used with
   refs and defs (ALLOC is the next place to allocate tabulated values). */

extern struct gen_binding Domains[] ;

/* Action parameter to the LOOKUP() function in the symbol table. Are 
   looking for a new symbol or an old one ? */

#define NEW_BINDING 1
#define OLD_BINDING 2

extern struct gen_binding *lookup() ;
extern struct gen_binding *new_binding() ;

/* Used to check, while parsing specs, that a constructed domain use 
   only one operator type. */

extern int Current_op ;

/* To manage imported domains:

   - NUMBER_IMPORTS is the number of imported domains (for which no
     access functions have to be generated - see TYPE in genC.c)
   - CURRENT_START is the beginning index (after TABULATED_BP) in Domains
   - CURRENT_FIRST is, for each OR_OP domain, the first tag number (used
     only in READ_SPEC_MODE */

extern int Number_imports ;
extern int Current_start ;
extern int Current_first ;
extern int Read_spec_mode ;

/* For tabulated domains, their index in the Gen_tabulated_ table. */

extern int Current_index ;
extern hash_table Gen_tabulated_names ;

/* For tabulated objects, the offset HASH_OFFSET of the hashed subdomain 
   and the separation character HASH_SEPAR between domain number and 
   hashed name. */

#define HASH_OFFSET 2
#define HASH_SEPAR '|'

/* External routines. */

/* extern char *strcpy() ; bb, 92.06.24 */
/* extern float atof() ; fi */
extern char *alloc() ;
extern void user() ;
#ifdef flex_scanner
extern char *zztext ;
extern char *xxtext ;
#else
extern char zztext[] ;
extern char xxtext[] ;
#endif
extern void gencode() ;
extern void fatal() ;
extern char *itoa() ;
extern void print_domains() ;
extern void init() ;
extern void compile() ;
/* extern char *malloc() ; bb, 92.06.24 */
/* FI: for genClib.c */
extern int gen_size();
extern int zzparse();
extern int xxparse();
extern void print_domain();
/* FI: for list.c */
extern void shared_pointers();
extern void gen_free_with_sharing();
