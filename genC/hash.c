/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@cri.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/

/* $RCSfile: hash.c,v $ ($Date: 1997/12/10 13:59:22 $, )
 * version $Revision$
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>

#include "newgen_types.h"
#include "genC.h"
#include "newgen_include.h"
#include "newgen_hash.h"

/* Some predefined values for the key 
 */

#define HASH_ENTRY_FREE (char *) 0
#define HASH_ENTRY_FREE_FOR_PUT (char *) -1

typedef struct __hash_entry hash_entry;

struct __hash_entry {
    char *key;
    char *val;
};

struct __hash_table {
    hash_key_type hash_type;
    int hash_size;
    int hash_entry_number;
    int (*hash_rank)();
    int (*hash_equal)();
    hash_entry *hash_array;
    int hash_size_limit;
};

#ifndef abs
#define abs(v) (((v) > 0) ? (v) : (-(v)))
#endif

/* Constant to find the end of the prime numbers table 
 */
#define END_OF_SIZE_TABLE ((int) 0)

/* >>1: 50 percent limit 
 * >>1 + >>2: 75 percent limit
 */
#define HASH_SIZE_LIMIT(size) (((size)>>1)+((size)>>2))

/* Hash function to get the index
 * of the array from the key 
 */
#define HASH_FUNCTION(key, size) ((((unsigned int)(key))&(0x7fffffff))%(size))

/* define the increment of the hash_function
 * in case of heat.
 * The old version can be achieved by setting
 * this macro to ((int) 1)
 */
#define HASH_FUNCTION_INCREMENT(key, size) \
    (1 + (((unsigned int)(key)&0x7fffffff)%((size) - 1)))

/* Now we need the table size to be a prime number.
 * So we need to retrieve the next prime number in a list.
 */
#define GET_NEXT_HASH_TABLE_SIZE(sz,pointer_to_table)			\
{									\
     while (*(pointer_to_table) <= (sz)) {				\
	 message_assert("size too big ",				\
			*(pointer_to_table) != END_OF_SIZE_TABLE);	\
	 (pointer_to_table)++;						\
     }									\
 (sz) = *(pointer_to_table);						\
}

/* Set of the different operations 
 */
typedef enum 
    { hash_get_op , hash_put_op , hash_del_op } hash_operation;

/* Private functions 
 */
static void hash_enlarge_table();
static hash_entry_pointer hash_find_entry();
static int hash_int_equal();
static int hash_int_rank();
static int hash_pointer_equal();
static int hash_pointer_rank();
static char *hash_print_key();
static int hash_string_equal();
static int hash_string_rank();
static int hash_chunk_equal() ;
static int hash_chunk_rank() ;

/* List of the prime numbers from 17 to 2^31-1 
 */
static int prime_numbers_for_table_size[] = {
    7,17,37,71,137,277,547,1091,2179,4357,8707,17417,
    34819,69653,139267,278543,557057,1114117,2228243,
    4456451,8912921,17825803,35651593,71303171,
    142606357,285212677,570425377,1140850699,
    END_OF_SIZE_TABLE};

/* internal variable to know we should warm or not
 */
static bool should_i_warn_on_redefinition = TRUE;

/* these function set the variable should_i_warn_on_redefinition
   to the value TRUE or FALSE */

void hash_warn_on_redefinition()
{
    should_i_warn_on_redefinition = TRUE;
}

void hash_dont_warn_on_redefinition()
{
    should_i_warn_on_redefinition = FALSE;
}

bool hash_warn_on_redefinition_p()
{
    return(should_i_warn_on_redefinition);
}

/* this function makes a hash table of size size. if size is less or
   equal to zero a default size is used. the type of keys is given by
   key_type (see hash.txt for further details). */

hash_table hash_table_make(key_type, size)
hash_key_type key_type;
int size;
{
    register int i;
    hash_table htp;
    int *prime_list = &prime_numbers_for_table_size[0];

    if (size<HASH_DEFAULT_SIZE) size=HASH_DEFAULT_SIZE - 1;
    /* get the next prime number in the table */
    GET_NEXT_HASH_TABLE_SIZE(size,prime_list);

    htp = (hash_table) alloc(sizeof(struct __hash_table));
    htp->hash_type = key_type;
    htp->hash_size = size;
    htp->hash_entry_number = 0;
    htp->hash_size_limit = HASH_SIZE_LIMIT(size);
    htp->hash_array = (hash_entry_pointer) alloc(size*sizeof(hash_entry));

    for (i = 0; i < size; i++) 
	htp->hash_array[i].key = HASH_ENTRY_FREE;
    
    switch(key_type)
    {
    case hash_string:
	htp->hash_equal = hash_string_equal;
	htp->hash_rank = hash_string_rank;
	break;
    case hash_int:
	htp->hash_equal = hash_int_equal;
	htp->hash_rank = hash_int_rank;
	break;
    case hash_chunk:
	htp->hash_equal = hash_chunk_equal;
	htp->hash_rank = hash_chunk_rank;
	break;
    case hash_pointer:
	htp->hash_equal = hash_pointer_equal;
	htp->hash_rank = hash_pointer_rank;
	break;
    default:
	fprintf(stderr, "[make_hash_table] bad type %d\n", key_type);
	abort();
    }

    return(htp);
}

/* Clears all entries of a hash table HTP. [pj] */

void hash_table_clear(htp)
hash_table htp;
{
    static int max_size_seen = 0;
    register hash_entry_pointer p ;
    register hash_entry_pointer end ;

    if (htp->hash_size > max_size_seen) {
	max_size_seen = htp->hash_size;
#ifdef DBG_HASH
	fprintf(stderr, "[hash_table_clear] maximum size is %d\n", 
		max_size_seen);
#endif
    }

    end = htp->hash_array + htp->hash_size ;
    htp->hash_entry_number = 0 ;

    for ( p = htp->hash_array ; p < end ; p++ ) {
	p->key = HASH_ENTRY_FREE ;
    }
}

/* this function deletes a hash table that is no longer useful. unused
 memory is freed. */

void hash_table_free(htp)
hash_table htp;
{
  free(htp->hash_array);
  free(htp);
}

/* This functions stores a couple (key,val) in the hash table pointed to
   by htp. If a couple with the same key was already stored in the table
   and if hash_warn_on_redefintion was requested, hash_put complains but
   replace the old value by the new one. This is a potential source for a
   memory leak. If the value to store is HASH_UNDEFINED_VALUE or if the key
   is HASH_ENTRY_FREE or HASH_ENTRY_FREE_FOR_INPUT, hash_put
   aborts. The restrictions on the key should be avoided by changing
   the implementation. The undefined value should be
   user-definable. It might be argued that users should be free to
   assign HASH_UNDEFINED_VALUE, but they can always perform hash_del()
   to get the same result */

void hash_put(htp, key, val)
hash_table htp;
char *key, *val;
{
    int rank;
    hash_entry_pointer hep;
    
    if (htp->hash_entry_number+1 >= (htp->hash_size_limit)) 
	hash_enlarge_table(htp);

    message_assert("illegal input key", key!=HASH_ENTRY_FREE &&
		   key!=HASH_ENTRY_FREE_FOR_PUT);
    message_assert("illegal input value", val!=HASH_UNDEFINED_VALUE);

    hep = hash_find_entry(htp, key, &rank, hash_put_op);
    
    if (hep->key != HASH_ENTRY_FREE && hep->key != HASH_ENTRY_FREE_FOR_PUT) {
	if (should_i_warn_on_redefinition && hep->val != val) {
	    (void) fprintf(stderr, "[hash_put] key redefined: %s\n", 
		    hash_print_key(htp->hash_type, key));
	}
	hep->val = val;
    }
    else {
	htp->hash_entry_number += 1;
	hep->key = key;
	hep->val = val;
    }
}

/* deletes key from the hash table. returns the val and key
 */ 
char * 
hash_delget(
    hash_table htp, 
    char * key, 
    char ** pkey)
{
    hash_entry_pointer hep;
    char *val;
    int rank;
    
    message_assert("legal input key",
		   key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT);

    hep = hash_find_entry(htp, key, &rank, hash_del_op);
    
    if (hep->key != HASH_ENTRY_FREE && hep->key != HASH_ENTRY_FREE_FOR_PUT) {
	val = hep->val;
	*pkey = hep->key;
	htp->hash_array[rank].key = HASH_ENTRY_FREE_FOR_PUT;
	htp->hash_entry_number -= 1;
	return val;
    }

    *pkey = 0;
    return HASH_UNDEFINED_VALUE;
}

/* this function removes from the hash table pointed to by htp the
   couple whose key is equal to key. nothing is done if no such couple
   exists. ??? shoudl abort ? (FC) */

char *hash_del(htp, key)
hash_table htp;
char *key;
{
    char * tmp;
    return hash_delget(htp, key, &tmp);
}

/* this function retrieves in the hash table pointed to by htp the
   couple whose key is equal to key. the HASH_UNDEFINED_VALUE pointer is
   returned if no such couple exists. otherwise the corresponding value
   is returned. */ 

char *hash_get(htp, key)
hash_table htp;
char *key;
{
    hash_entry_pointer hep;
    int n;

    message_assert("legal input key", key!=HASH_ENTRY_FREE &&
		   key!=HASH_ENTRY_FREE_FOR_PUT);

    if (!htp->hash_entry_number) 
	return HASH_UNDEFINED_VALUE;

    /* else may be there */
    hep = hash_find_entry(htp, key, &n, hash_get_op);
    
    return(hep->key!=HASH_ENTRY_FREE && hep->key!=HASH_ENTRY_FREE_FOR_PUT ? 
	   hep->val : HASH_UNDEFINED_VALUE);
}

/* TRUE if key has e value in htp.
 */
bool hash_defined_p(htp, key)
hash_table htp;
char *key;
{
    return(hash_get(htp, key)!=HASH_UNDEFINED_VALUE);
}

/* update key->val in htp, that MUST be pre-existent.
 */
void hash_update(htp, key, val)
hash_table htp;
char *key, *val;
{
    hash_entry_pointer hep;
    int n;

    message_assert("illegal input key", key!=HASH_ENTRY_FREE &&
		   key!=HASH_ENTRY_FREE_FOR_PUT);
    hep = hash_find_entry(htp, key, &n, hash_get_op);
    message_assert("no previous entry", htp->hash_equal(hep->key, key));
    
    hep->val = val ;
}

/* this function prints the header of the hash_table pointed to by htp
 * on the opened stream fout 
 */
void hash_table_print_header(htp,fout)
hash_table htp;
FILE *fout;
{
    fprintf(fout, "hash_key_type:     %d\n", htp->hash_type);
    fprintf(fout, "hash_size:         %d\n", htp->hash_size);
    /* to be used by pips, we should not print this
       as it is only for debugging NewGen and it is not important data
       I (go) comment it.
       
       fprintf(fout, "hash_size_limit    %d\n", htp->hash_size_limit);
       */
    fprintf(fout, "hash_entry_number: %d\n", htp->hash_entry_number);
}
 
/* this function prints the content of the hash_table pointed to by htp
on stderr. it is mostly useful when debugging programs. */

void hash_table_print(htp)
hash_table htp;
{
    int i;

    hash_table_print_header (htp,stderr);

    for (i = 0; i < htp->hash_size; i++) {
	hash_entry he;

	he = htp->hash_array[i];

	if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
	    fprintf(stderr, "%d %s %x\n", 
		    i, hash_print_key(htp->hash_type, he.key),
		    (unsigned int) he.val);
	}
    }
}

/* This function prints the content of the hash_table pointed to by htp
on file descriptor f, using functions key_to_string and value_to string
to display the mapping. it is mostly useful when debugging programs. */

void hash_table_fprintf(f, key_to_string, value_to_string, htp)
FILE * f;
char * (*key_to_string)();
char * (*value_to_string)();
hash_table htp;
{
    int i;

    hash_table_print_header (htp,f);

    for (i = 0; i < htp->hash_size; i++) {
	hash_entry he;

	he = htp->hash_array[i];

	if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
	    fprintf(f, "%s -> %s\n", 
		    key_to_string(he.key), value_to_string(he.val));
	}
    }
}

/* function to enlarge the hash_table htp.
 * the new size will be first number in the array prime_numbers_for_table_size
 * that will be greater or equal to the actual size 
 */

static void 
hash_enlarge_table(hash_table htp)
{
    hash_entry_pointer old_array;
    int i, old_size;
    int *prime_list = &prime_numbers_for_table_size[0];

    old_size = htp->hash_size;
    old_array = htp->hash_array;

    htp->hash_size++;
    /* Get the next prime number in the table */
    GET_NEXT_HASH_TABLE_SIZE(htp->hash_size,prime_list);
    htp->hash_array = (hash_entry_pointer) 
	malloc(htp->hash_size* sizeof(hash_entry));
    htp->hash_size_limit = HASH_SIZE_LIMIT(htp->hash_size);

    for (i = 0; i < htp->hash_size ; i++)
	htp->hash_array[i].key = HASH_ENTRY_FREE;

    for (i = 0; i < old_size; i++) 
    {
	hash_entry he;
	he = old_array[i];

	if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
	    hash_entry_pointer nhep;
	    int rank;

	    nhep = hash_find_entry(htp, he.key, &rank, hash_put_op);

	    if (nhep->key != HASH_ENTRY_FREE) {
		fprintf(stderr, "[hash_enlarge_table] fatal error\n");
		abort();
	    }    
	    htp->hash_array[rank] = he;
	}
    }
    free(old_array);
}

static int hash_string_rank(key, size)
char *key;
int size;
{
    int v;
    char *s;

    v = 0;
    for (s = key; *s; s++)
 	v <<= 2, v += *s;
    v = abs(v) ;
    v %= size ;

    return v;
}

static int hash_int_rank(key, size)
char *key;
int size;
{
    return HASH_FUNCTION(key, size);
}

static int hash_pointer_rank(key, size)
char *key;
int size;
{
    return HASH_FUNCTION(key, size);
}

static int hash_chunk_rank(key, size)
char *key;
int size;
{
    return HASH_FUNCTION(((chunk *)key)->i, size);
}

static int hash_string_equal(key1, key2)
char *key1, *key2;
{
    return strcmp(key1, key2)==0;
}

static int hash_int_equal(key1, key2)
char *key1, *key2;
{
    return key1 == key2;
}

static int hash_pointer_equal(key1, key2)
char *key1, *key2;
{
    return key1 == key2;
}

static int hash_chunk_equal(key1, key2)
gen_chunk *key1, *key2;
{
    return memcmp((char*)key1, (char*)key2, sizeof(gen_chunk))==0;
}

static char *hash_print_key(t, key)
hash_key_type t;
char *key;
{
    static char buffer[256];

    if (t == hash_string)
	sprintf(buffer, "%s", key);
    else if (t == hash_int)
	sprintf(buffer, "%d", (int)key);
    else if (t == hash_pointer)
	sprintf(buffer, "%x", (unsigned int) key);
    else if (t == hash_chunk)
	sprintf(buffer, "%x", (unsigned int)((gen_chunk *)key)->p);	    
    else {
	fprintf(stderr, "[hash_print_key] bad type : %d\n", t);
	abort();
    }

    return(buffer);
}

/*  buggy function, the hash table stuff should be made again from scratch.
 *  - FC 02/02/95
 */
static hash_entry_pointer hash_find_entry(htp, key, prank, operation)
hash_table htp;
char *key;
int *prank;
hash_operation operation;
{
    int r;
    hash_entry he;
    int r_init ;
    int r_increment;

    r_init = r = (*(htp->hash_rank))(key, htp->hash_size);
    r_increment = HASH_FUNCTION_INCREMENT(r_init, htp->hash_size);

    while (1) {
	he = htp->hash_array[r];

	if (he.key == HASH_ENTRY_FREE)
	    break;
	
	/*  ??? it may happen that the previous mapping is kept
	 *  somewhere forward! So after a hash_del, the old value
	 *  would be visible again!
	 */
	if (he.key == HASH_ENTRY_FREE_FOR_PUT && operation == hash_put_op)
	    break;

	if (he.key != HASH_ENTRY_FREE_FOR_PUT &&
	    (*(htp->hash_equal))(he.key, key))
	    break;

	/* GO: it is not anymore the next slot
	 * we skip some of them depending on the
	 * reckonned increment 
	 */
	r = (r + r_increment) % htp->hash_size;

	/*   ??? this may happen in a hash_get after many put and del,
	 *   if the table contains no FREE, but many FREE_FOR_PUT instead!
	 */
	if( r == r_init ) {
	    fprintf(stderr,"[hash_find_entry] cannot find entry\n") ;
	    abort() ;
	}
    }
    *prank = r;
    return(&(htp->hash_array[r]));
}

/* now we define observers in order to
 * hide the hash_table type ... 
 */
int hash_table_entry_count(htp)
hash_table htp;
{
    return htp->hash_entry_number;
}

int hash_table_size(htp)
hash_table htp;
{
    return htp->hash_size;
}

hash_key_type hash_table_type(htp)
hash_table htp;
{
    return htp->hash_type;
}

/*
 * This function allows a hash_table scanning
 * First you give a NULL hentryp and get the key and val
 * After you give the previous hentryp and so on
 * at the end NULL is returned
 */

hash_entry_pointer hash_table_scan(htp, hentryp, pkey, pval)
hash_table htp;
hash_entry_pointer hentryp;
char **pkey;
char **pval;
{
    hash_entry_pointer hend = htp->hash_array + htp->hash_size;

    if (!hentryp)
	hentryp = htp->hash_array;

    while (hentryp < hend)
    {
	char *key = hentryp->key;

	if ((key !=HASH_ENTRY_FREE) && (key !=HASH_ENTRY_FREE_FOR_PUT))
	{
	    *pkey = key;
	    *pval = hentryp->val;
	    return hentryp + 1;
	}
	hentryp++;
    }
    return NULL;
}

int
hash_table_own_allocated_memory(
    hash_table htp)
{
    return htp ? 
      sizeof(struct __hash_table) + sizeof(hash_entry)*(htp->hash_size) : 0 ;
}
