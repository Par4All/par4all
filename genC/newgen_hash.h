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

/* $RCSfile: newgen_hash.h,v $ ($Date: 1995/04/06 17:12:31 $, )
 * version $Revision$
 * got on %D%, %T%
 */

#ifndef HASH_INCLUDED
#define HASH_INCLUDED
#define HASH_DEFAULT_SIZE 17

/* Some predefined values for the key 
 */
#define HASH_ENTRY_FREE (char *) 0
#define HASH_ENTRY_FREE_FOR_PUT (char *) -1

typedef enum hash_key_type { 
    hash_string, hash_int, hash_pointer, hash_chunk } hash_key_type;

typedef struct hash_entry {
    char *key;
    char *val;
} hash_entry, *hash_entry_pointer;

typedef struct hash_table {
    hash_key_type hash_type;
    int hash_size;
    int hash_entry_number;
    int (*hash_rank)();
    int (*hash_equal)();
    hash_entry *hash_array;
    int hash_size_limit;
} *hash_table;

#define hash_table_undefined ((hash_table)gen_chunk_undefined)  

/* value returned by hash_get() when the key is not found; could also be
   called HASH_KEY_NOT_FOUND, but it's semantically a value; this bottom
   value will be user-definable in a future release of NewGen */

#define HASH_UNDEFINED_VALUE ((char *) gen_chunk_undefined)

#define hash_table_empty_p(htp) (hash_table_count(htp) == 0)

#define HASH_MAP(k,v,code,h) \
    {\
    hash_table _map_hash_h = (h) ;\
    register hash_entry_pointer _map_hash_p = hash_table_array(_map_hash_h) ;\
    hash_entry_pointer _map_hash_end = \
	    hash_table_array(_map_hash_h)+hash_table_size(_map_hash_h) ;\
    for( ; _map_hash_p<_map_hash_end ; _map_hash_p++ ) { \
	if( hash_entry_key(_map_hash_p) !=HASH_ENTRY_FREE && \
            hash_entry_key(_map_hash_p) !=HASH_ENTRY_FREE_FOR_PUT) { \
	    char *k = hash_entry_key(_map_hash_p) ; \
	    char *v = hash_entry_val(_map_hash_p) ; \
            code ; }}}
#endif

/* Let's define a new version of
 * hash_put_or_update() using the warn_on_redefinition 
 */

#define hash_put_or_update(h, k, v)			\
if (hash_warn_on_redefinition_p() == TRUE)		\
{							\
    hash_dont_warn_on_redefinition();			\
    hash_put((hash_table)h, (char*)k, (char*)v);	\
    hash_warn_on_redefinition();			\
} else							\
    hash_put((hash_table)h, (char*)k, (char*)v);

/*
#define hash_put_or_update(h, k, v)\
  if (hash_get((hash_table)h, (char*)k)==HASH_UNDEFINED_VALUE)\
    hash_put((hash_table)h, (char*)k, (char*)v);\
  else\
    hash_update((hash_table)h, (char*)k, (char*)v);
*/

/* functions declared in hash.c 
 */
extern void hash_warn_on_redefinition GEN_PROTO(());
extern void hash_dont_warn_on_redefinition GEN_PROTO(());
extern char *hash_del GEN_PROTO((hash_table, char *));
extern char *hash_get GEN_PROTO((hash_table, char *));
extern bool hash_defined_p GEN_PROTO((hash_table, char *));
extern void hash_put GEN_PROTO((hash_table, char *, char *));
extern void hash_table_clear GEN_PROTO((hash_table));
extern void hash_table_free GEN_PROTO((hash_table));
extern hash_table hash_table_make GEN_PROTO((hash_key_type, int));
extern void hash_table_print_header GEN_PROTO((hash_table, FILE *));
extern void hash_table_print GEN_PROTO((hash_table));
extern void hash_table_fprintf GEN_PROTO((FILE *, char *(*)(), 
					  char *(*)(), hash_table));
extern void hash_update GEN_PROTO((hash_table, char*, char*));

extern int hash_table_entry_count GEN_PROTO((hash_table));
extern int hash_table_size GEN_PROTO((hash_table));
extern hash_key_type hash_table_type GEN_PROTO((hash_table));
extern hash_entry_pointer hash_table_array GEN_PROTO((hash_table));
extern char *hash_entry_val GEN_PROTO((hash_entry_pointer));
extern char *hash_entry_key GEN_PROTO((hash_entry_pointer));


/*  that is all
 */
