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


#include <stdio.h>
extern int fprintf();
#include <string.h>
#include <stdlib.h>
#include "newgen_types.h"
#include "genC.h"
#include "newgen_include.h"
#include "newgen_hash.h"
extern int cfree();


#define abs(v) (((v) > 0) ? (v) : (-(v)))

#define FREE (char *) 0
#define FREE_FOR_PUT (char *) -1

/* >>1: 50 percent limit 
 * >>1 + >>2: 75 percent limit
 */
#define HASH_SIZE_LIMIT(size) (((size)>>1)+((size)>>2))
#define HASH_ENLARGE_PARAMETER ((int)2)
#define HASH_FUNCTION(key, size) ((((int)(key))&(0x7fffffff))%(size))

typedef enum hash_operation 
    { hash_get_op , hash_put_op , hash_del_op } hash_operation;

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

static bool should_i_warn_on_redefinition = TRUE;

/* these function set the variable should_i_warn_on_redefinition to the 
value TRUE or FALSE */

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
    return( should_i_warn_on_redefinition ) ;
}

/* this function makes a hash table of size size. if size is less or
equal to zero a default size is used. the type of keys is given by
key_type (see hash.txt for further details).  */

hash_table hash_table_make(key_type, size)
hash_key_type key_type;
int size;
{
    register int i;
    extern int fprintf();

    if (size <= 0) size = HASH_DEFAULT_SIZE;

    if (size == 1) 
    {
	fprintf(stderr, "[hash_table_make] size %d too small\n", size);
	exit(1);
    }	    

    if (key_type==hash_string||
	key_type==hash_int||
	key_type==hash_pointer||
	key_type==hash_chunk) {
	hash_table htp;

	htp = (hash_table) alloc(sizeof(struct hash_table));
	htp->hash_type = key_type;
	htp->hash_size = size;
	htp->hash_entry_number = 0;
	htp->hash_size_limit = HASH_SIZE_LIMIT(size);

	htp->hash_array = (hash_entry *) alloc( size*sizeof(hash_entry));
	for (i = 0; i < size; i++)
		htp->hash_array[i].key = FREE;

	if (key_type == hash_string) {
	    htp->hash_equal = hash_string_equal;
	    htp->hash_rank = hash_string_rank;
	}
	else if (key_type == hash_int) {
	    htp->hash_equal = hash_int_equal;
	    htp->hash_rank = hash_int_rank;
	}
	else if (key_type == hash_chunk) {
	    htp->hash_equal = hash_chunk_equal;
	    htp->hash_rank = hash_chunk_rank;
	}
	else {
	    htp->hash_equal = hash_pointer_equal;
	    htp->hash_rank = hash_pointer_rank;
	}

	return(htp);
    }

    fprintf(stderr, "[make_hash_table] bad type %d\n", key_type);
    abort();
}

/* Clears all entries of a hash table HTP. [pj] */

void hash_table_clear(htp)
hash_table htp;
{
    extern int fprintf();
    static int max_size_seen = 0;
    register hash_entry *p ;
    register hash_entry *end ;

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
	p->key = FREE ;
    }
}

/* this function deletes a hash table that is no longer useful. unused
memory is freed.  */

void hash_table_free(htp)
hash_table htp;
{
    cfree(htp->hash_array);
    free(htp);
}

/* This functions stores a couple (key,val) in the hash table pointed to
   by htp. If a couple with the same key was already stored in the table
   and if hash_warn_on_redefintion was requested, hash_put complains but
   replace the old value by the new one. This is a potential source for a
   memory leak. If the value to store is HASH_UNDEFINED_VALUE or if the key
   is FREE or FREE_FOR_INPUT, hash_put aborts. The restrictions on the key
   should be avoided by changing the implementation. The undefined value
   should be user-definable. It might be argued that users should be free
   to assign HASH_UNDEFINED_VALUE, but they can always perform hash_del()
   to get the same result */

void hash_put(htp, key, val)
hash_table htp;
char *key, *val;
{
    extern int fprintf();
    int rank;
    hash_entry_pointer hep;
    
    if (htp->hash_entry_number+1 >= (htp->hash_size_limit)) 
	hash_enlarge_table(htp);

    if( key == FREE || key == FREE_FOR_PUT ) {
	user( "hash_put: illegal input key\n", "" ) ;
	abort() ;
    }
    if( val == HASH_UNDEFINED_VALUE ) {
	user( "hash_put: illegal input value\n", "" ) ;
	abort() ;
    }
    hep = hash_find_entry(htp, key, &rank, hash_put_op);
    
    if (hep->key != FREE && hep->key != FREE_FOR_PUT) {
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

/* this function removes from the hash table pointed to by htp the
   couple whose key is equal to key. nothing is done if no such couple
   exists. */

char *hash_del(htp, key)
hash_table htp;
char *key;
{
    hash_entry_pointer hep;
    char *val;
    int rank;
    
    hep = hash_find_entry(htp, key, &rank, hash_del_op);
    
    if( key == FREE || key == FREE_FOR_PUT ) {
	user( "hash_del: illegal input key\n", "" ) ;
	abort() ;
    }
    if (hep->key != FREE && hep->key != FREE_FOR_PUT) {
	val = hep->val;
	htp->hash_array[rank].key = FREE_FOR_PUT;
	htp->hash_entry_number -= 1;
	return(val);
    }

    return(HASH_UNDEFINED_VALUE);
}

/* this function retrieves in the hash table pointed to by htp the
   couple whose key is equal to key. the HASH_UNDEFINED_VALUE pointer is
   returned if no such couple exists. otherwise the corresponding value
   is returned. */ 

char *hash_get(htp, key)
hash_table htp;
char *key;
{
    int n;
    hash_entry_pointer hep;

    if( key == FREE || key == FREE_FOR_PUT ) {
	user( "hash_get: illegal input key\n", "" ) ;
	abort() ;
    }
    hep = hash_find_entry(htp, key, &n, hash_get_op);
    
    if (hep->key != FREE && hep->key != FREE_FOR_PUT) {
	return(hep->val);
    }
    else {
	return(HASH_UNDEFINED_VALUE);
    }
}

/* this function updates in the hash table pointed to by htp the
   couple whose key is equal to key with val (which is returned). */

char *hash_update(htp, key, val)
hash_table htp;
char *key, *val;
{
    int n;
    hash_entry_pointer hep;

    if( key == FREE || key == FREE_FOR_PUT ) {
	user( "hash_update: illegal input key\n", "" ) ;
	abort() ;
    }
    hep = hash_find_entry(htp, key, &n, hash_get_op);
    
    if (hep->key != FREE && hep->key != FREE_FOR_PUT) {
	hep->val = val ;
    }
    else {
	user( "hash_update: input key not in hash table\n", "" ) ;
    }
    return( val );
}

/* this function prints the content of the hash_table pointed to by htp
on stderr. it is mostly useful when debugging programs. */

void hash_table_print(htp)
hash_table htp;
{
    extern int fprintf();
    int i;

    fprintf(stderr, "hash_key_type:     %d\n", htp->hash_type);
    fprintf(stderr, "hash_size:         %d\n", htp->hash_size);
    fprintf(stderr, "hash_size_limit    %d\n", htp->hash_size_limit);
    fprintf(stderr, "hash_entry_number: %d\n", htp->hash_entry_number);

    for (i = 0; i < htp->hash_size; i++) {
	hash_entry he;

	he = htp->hash_array[i];

	if (he.key != FREE && he.key != FREE_FOR_PUT) {
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
    extern int fprintf();
    int i;

    fprintf(f, "hash_key_type:     %d\n", htp->hash_type);
    fprintf(f, "hash_size:         %d\n", htp->hash_size);
    fprintf(f, "hash_size_limit    %d\n", htp->hash_size_limit);
    fprintf(f, "hash_entry_number: %d\n", htp->hash_entry_number);

    for (i = 0; i < htp->hash_size; i++) {
	hash_entry he;

	he = htp->hash_array[i];

    if (he.key != FREE && he.key != FREE_FOR_PUT) {
	    fprintf(f, "%s -> %s\n", 
		    key_to_string(he.key), value_to_string(he.val));
	}
    }
}

int hash_table_entry_count(htp)
hash_table htp;
{
    return htp->hash_entry_number;
}

static void hash_enlarge_table(htp)
hash_table htp;
{
    extern int fprintf();
    hash_entry *old_array;
    int i, old_size;

    old_size = htp->hash_size;
    old_array = htp->hash_array;

    htp->hash_size *= HASH_ENLARGE_PARAMETER ;
    htp->hash_array = 
	    (hash_entry *) alloc( htp->hash_size* sizeof(hash_entry));
    htp->hash_size_limit = HASH_SIZE_LIMIT(htp->hash_size);

    for (i = 0; i < htp->hash_size ; i++) {
	htp->hash_array[i].key = FREE;
    }
    for (i = 0; i < old_size; i++) {
	hash_entry he;

	he = old_array[i];

	if (he.key != FREE && he.key != FREE_FOR_PUT) {
	    hash_entry_pointer nhep;
	    int rank;

	    nhep = hash_find_entry(htp, he.key, &rank, hash_put_op);

	    if (nhep->key != FREE) {
		fprintf(stderr, "[hash_enlarge_table] fatal error\n");
		abort();
	    }    
	    htp->hash_array[rank] = he;
	}
    }
    cfree(old_array);
}

static int hash_string_rank(key, size)
char *key;
int size;
{
    int v;
    char *s;

    v = 0;
    for (s = key; *s; s++) {
	v += *s;
	v <<= 2 ;
    }
    v = abs( v ) ;
    v %= size ;

    return(v);
}

static int hash_int_rank(key, size)
char *key;
int size;
{
/*
    int v;

    v = abs((int)key) ;
    v %= size;

    return(v);
*/
    return(HASH_FUNCTION(key, size));
}

static int hash_pointer_rank(key, size)
char *key;
int size;
{
/*
    unsigned int skey;
    int v;

    skey = (unsigned int) key;
    v = abs((int) (skey>>2));
    v %= size;

    return(v);
*/

    return(HASH_FUNCTION(key, size));
}

static int hash_chunk_rank(key, size)
char *key;
int size;
{
/*
    unsigned int skey;
    int v;

    skey = (unsigned int) ((chunk *)key)->i ;
    v = abs((int) (skey>>2)); 
    v %= size;

    return(v);
*/
    return(HASH_FUNCTION(key, size));
}

static int hash_string_equal(key1, key2)
char *key1, *key2;
{
    return(strcmp(key1, key2) == 0);
}

static int hash_int_equal(key1, key2)
char *key1, *key2;
{
    return( (int)key1 == (int)key2 );
}

static int hash_pointer_equal(key1, key2)
char *key1, *key2;
{
    return( key1 == key2);
}

static int hash_chunk_equal(key1, key2)
gen_chunk *key1, *key2;
{
    return(memcmp(key1, key2, sizeof(gen_chunk)) == 0) ;
}

static char *hash_print_key(t, key)
hash_key_type t;
char *key;
{
    extern int fprintf();
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

static hash_entry_pointer hash_find_entry(htp, key, prank, operation)
hash_table htp;
char *key;
int *prank;
hash_operation operation;
{
    extern int fprintf();
    int r;
    hash_entry he;
    int r_init ;

    r_init = r = (*(htp->hash_rank))(key, htp->hash_size);

    while (1) {
	he = htp->hash_array[r];

	if (he.key == FREE)
	    break;

	if (he.key == FREE_FOR_PUT && operation == hash_put_op)
	    break;

	if (he.key != FREE_FOR_PUT && (*(htp->hash_equal))(he.key, key))
	    break;

	r = (r == htp->hash_size-1) ? 0 : r+1;

	if( r == r_init ) {
	    fprintf(stderr,"[hash_find_entry] cannot find entry\n") ;
	    abort() ;
	}
    }
    *prank = r;
    return(&(htp->hash_array[r]));
}

