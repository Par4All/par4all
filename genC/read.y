%{
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


/* read.y 

   The syntax of objects printed by GEN_WRITE. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "newgen_include.h"

extern int gen_find_free_tabulated(struct gen_binding *);
extern void newgen_lexer_position(FILE *);

#define YYERROR_VERBOSE 1 /* better error messages by bison */

extern int yyinput(void);
extern void yyerror(char*);

extern FILE * yyin;

/* This constant should be adapted to the particular need of the application */

/* set to 10000 by BC - necessary in PIPS for DYNA */
/* Should be a compilation option ? */
/* CA: pb avec COX si a 10000... p'tet mauvaise recursion dans le parser de newgen? */
#define YYMAXDEPTH 100000

/* User selectable options. */

int warn_on_ref_without_def_in_read_tabulated = FALSE;

/* Where the root will be. */

gen_chunk *Read_chunk ;

/* The SHARED_TABLE maps a shared pointer number to its gen_chunk pointer value. */

static gen_chunk **shared_table ;
static int shared_number ;

/* The GEN_TABULATED_NAMES hash table maps ids to index in the table of
   the tabulated domains. In case of multiple definition, if the previous
   value is negative, then it came from a REF (by READ_TABULATED) and
   no error is reported. */

/* Management of forward references in read */

int allow_forward_ref = FALSE ;

static char *read_external() ;
static gen_chunk *make_def(), *make_ref() ;

%}

%token CHUNK_BEGIN
%token VECTOR_BEGIN
%token ARROW_BEGIN
%token READ_BOOL
%token LP
%token RP
%token LC
%token RC
%token LB
%token SHARED_POINTER
%token READ_EXTERNAL
%token READ_DEF
%token READ_REF
%token READ_NULL

%token READ_LIST_UNDEFINED
%token READ_SET_UNDEFINED
%token READ_ARRAY_UNDEFINED
%token READ_STRING

%union {
  gen_chunk chunk ;
  gen_chunk *chunkp ;
  cons *consp ;
  int val ;
  char * s;
  double d;
  char c;
}

%term READ_UNIT
%term <c> READ_CHAR
%term <val> READ_INT
%term <d> READ_FLOAT
%type <s> READ_STRING
%type <chunk> Data Basis 
%type <chunkp> Chunk String
%type <consp> Sparse_Datas Datas
%type <val> Int Shared_chunk Type

%%
Read	: Nb_of_shared_pointers Chunk {
		Read_chunk = $2 ;
		free( shared_table ) ;
		YYACCEPT ;
                /*NOTREACHED*/
		}
	;

Nb_of_shared_pointers 
  	: Int 	{
		shared_table = (gen_chunk **)alloc( $1*sizeof( gen_chunk * )) ;
		}
	;

Chunk 	: Shared_chunk CHUNK_BEGIN Type Datas RP {
  		int i ;
		cons *cp ;
		int length = gen_length( $4 ) ;

		$$ = ($1) ? shared_table[ $1-1 ]:
			(gen_chunk *)alloc( (GEN_HEADER+length)*sizeof( gen_chunk )) ;
		$$->i = $3 ;

		for( i=0, cp=gen_nreverse( $4 ); i<length ; i++, cp = cp->cdr )
			*($$+1+i) = cp->car ;
#ifdef DBG_READ
		write_chunk( stderr, $$, GEN_HEADER+length ) ;
#endif
		}
	;

Shared_chunk
	: LB Int {
		$$ = shared_number = $2 ;
		}
	|	{
		$$ = shared_number = 0 ;
		}
	;

Type	: Int 
	{
	  int type_number = gen_type_translation_old_to_actual($1);
	  if( shared_number ) {
	    struct gen_binding *bp = &Domains[ type_number ] ;
	    
	    shared_table[ shared_number-1 ] = 
	      (gen_chunk *)alloc(gen_size( bp )*
				 sizeof( gen_chunk )) ;
	  }
	  $$ = type_number ;
	}
	;

Datas	: Datas Data {	
	        $$ = CONS( CHUNK, $2.p, $1 ) ;
		}
	| 	{
		$$ = NIL ;
		}
	;

Sparse_Datas	
	: Sparse_Datas Int Data {	
	        $$ = CONS(CONSP, 
			  CONS( INT, $2, CONS( CHUNK, $3.p, NIL)), 
			  $1 ) ;
		}
	| 	{
		$$ = NIL ;
		}
	;

Data	: Basis	{
                $$ = $1 ;
		}
        | READ_LIST_UNDEFINED {
	        $$.l = list_undefined ;
	        }
	| LP Datas RP {
		$$.l = gen_nreverse( $2 ) ;
		}
        | READ_SET_UNDEFINED {
	        $$.t = set_undefined ;
	    }
        | LC Int Datas RC {
	        $$.t = set_make( $2 ) ;
		MAPL( cp, {
		  switch( $2 ) {
		  case set_int:
		    set_add_element( $$.t, $$.t, (char *)cp->car.i ) ;
		    break ;
		  default:
		    set_add_element( $$.t, $$.t, cp->car.s ) ;
		    break ;
		  }}, $3 ) ;
		gen_free_list( $3 ) ;
	        }
        | READ_ARRAY_UNDEFINED {
	        $$.p = array_undefined ;
	        }
	| VECTOR_BEGIN Int Sparse_Datas RP {
                gen_chunk *kp ;
		cons *cp ;
		int i ;

		kp = (gen_chunk *)alloc( $2*sizeof( gen_chunk )) ;

		for( i=0 ; i != $2 ; i++ ) {
		    kp[ i ].p = gen_chunk_undefined ;
		}
		for( cp=$3 ; cp!=NULL ; cp=cp->cdr ) {
		    cons *pair = CONSP( CAR( cp )) ;
		    
		    kp[ INT(CAR(pair)) ] = CAR(CDR(pair)) ;
		}
		gen_free_list( $3 ) ;
                $$.p = kp ;
		}
	| ARROW_BEGIN Datas RP {
		hash_table h = hash_table_make( hash_chunk, 0 )	;
		cons *cp ;

		for( cp = gen_nreverse($2) ; cp != NULL ; cp=cp->cdr->cdr ) {
			gen_chunk *k = (gen_chunk *)alloc(sizeof(gen_chunk));
			gen_chunk *v = (gen_chunk *)alloc(sizeof(gen_chunk));
	
			*k = CAR(  cp ) ;
			*v = CAR( CDR( cp )) ;
			hash_put( h, (char *)k, (char *)v ) ;
		}
		gen_free_list( $2 ) ;
		$$.h = h ;
		}
	| Chunk {
		$$.p = $1 ;
		}
	| SHARED_POINTER Int {
		$$.p = shared_table[ $2-1 ] ;
		}
	;
  
Basis	: READ_UNIT 	{
	        $$.u = 1 ;
		}
	| READ_BOOL Int {
		$$.b = $2 ;
		}
	| READ_CHAR	{
		$$.c = $1 ;
		}
	| Int	{
		$$.i = $1 ;
		}
	| READ_FLOAT {
		$$.f = $1 ;
		}
	| String {
	        $$ = *$1 ;
		}
 	| READ_EXTERNAL Int {
		$$.s = read_external( $2 ) ;
		}
	| READ_DEF Int String Chunk {
	        $$.p = make_def( $2, $3, $4 ) ;
	        }
	| READ_REF Int String {
	        $$.p = make_ref( $2, $3 ) ;
	        }
	| READ_NULL {
		$$.p = gen_chunk_undefined ;
		}
	;

Int     : READ_INT   {
  		$$ = $1 ;
		}
	;

String  : READ_STRING {
		gen_chunk *obj = (gen_chunk *)alloc(sizeof(gen_chunk));
		char * p;

		/* special management of string_undefined... FC. 12/95.
		 */
		if (disk_string_undefined_p($1)) {
		    free($1);
		    p = string_undefined;
		} else
		    p = $1;

		obj->s = p ;
		$$ = obj ;
	    }
		    
%%

/* YYERROR manages a syntax error while reading an object. */

void yyerror(char * s)
{
  int c, n=40;
  newgen_lexer_position(stderr);
  fprintf(stderr, "%s before ", s);

  while (n-->0  && ((c=yyinput()) != EOF))
    putc(c, stderr);

  fprintf(stderr, "\n\n");

  fatal("Incorrect object written by GEN_WRITE\n", (char *) NULL);
}

/* READ_EXTERNAL reads external types on stdin */

static char * read_external(int which)
{
    struct gen_binding *bp = &Domains[ which ] ;
    union domain *dp = bp->domain ;
    extern int yyinput() ;

    if( dp->ba.type != EXTERNAL_DT ) {
	fatal( "gen_read: undefined external %s\n", bp->name ) ;
	/*NOTREACHED*/
    }
    if( dp->ex.read == NULL ) {
	user( "gen_read: uninitialized external %s\n", bp->name ) ;
	return( NULL ) ;
    }
    if( yyinput() != ' ' ) {
	fatal( "read_external: white space expected\n", (char *)NULL ) ;
	/*NOTREACHED*/
    }
    /*
      Attention, ce qui suit est absolument horrible. Les fonctions
      suceptibles d'etre  appelees a cet endroit sont:
      - soit des fonctions 'user-written' pour les domaines externes
      non geres par NewGen
      - soit la fonctions gen_read pour les domaines externes geres
      par NewGen 
      
      Dans le 1er cas, il faut passer la fonction de lecture d'un caractere
      (yyinput) a la fonction de lecture du domaine externe (on ne peut pas
      passer le pointeur de fichier car lex bufferise les caracteres en
      entree). Dans le second cas, il faut passer le pointeur de fichier a
      cause de yacc/lex.
      
      Je decide donc de passer les deux parametres: pointeur de fichier et
      pointeur de fonction de lecture. Dans chaque cas, l'un ou l'autre sera
      ignore. 
      */
    return( (*(dp->ex.read))( yyin, yyinput )) ;
}


/* ENTER_TABULATED_DEF enters a new definition (previous refs are allowed if
   ALLOW_REF) in the INDEX tabulation table of the DOMAIN, with the unique
   ID and value CHUNKP. */

gen_chunk *
enter_tabulated_def(
    int index,
    int domain,
    char *id,
    gen_chunk *chunkp,
    int allow_ref)
{
    gen_chunk *hash ;

    if( Gen_tabulated_[ index ] == (gen_chunk *)NULL ) {
	fatal( "enter_tabulated_def: Uninitialized %s\n", 
	       Domains[ domain ].name ) ;
    }
    
    if ((hash=(gen_chunk *) gen_get_tabulated_name_basic(domain, id)) !=
	(gen_chunk *)HASH_UNDEFINED_VALUE ) {
	
	/* redefinitions of tabulated should not be allowed...
	 * but you cannot be user it is a redefinition if allow_ref
	 */
	if(!allow_ref)
	    (void) fprintf(stderr, "[make_%s] warning: %s redefined\n", 
			   Domains[domain].name, id);

	/* actually very obscure there... seems that negative domain
	 * numbers are used to encode something... already used/seen ???
	 */
	if( allow_ref && hash->i < 0 ) 
	{
	    int i, size = gen_size( Domains+domain ) ;
	    gen_chunk *cp, *gp ;

	    hash->i = -hash->i ;

	    if( (gp=(Gen_tabulated_[ index ]+hash->i)->p) == NULL ) {
		fatal( "make_def: Null for %d%c%s\n", domain, HASH_SEPAR, id);
	    }
	    for( cp=chunkp, i=0 ; i<size ; i++ ) {
		*gp++ = *cp++ ;
	    }
	    ((Gen_tabulated_[ index ]+hash->i)->p+1)->i = hash->i ;
	    return( (Gen_tabulated_[ index ]+hash->i)->p ) ;
	} 
	else {
	    if (hash_warn_on_redefinition_p()) {
		user("Tabulated entry %d%c%s already defined: updating\n",
		     domain, HASH_SEPAR, id);
	    }
	}
    }
    else {
	hash = (gen_chunk *)alloc( sizeof( gen_chunk )) ;
	hash->i = gen_find_free_tabulated( &Domains[ domain ] ) ;
	gen_put_tabulated_name(domain, id, (char *)hash);
    }
    (Gen_tabulated_[ index ]+hash->i)->p = chunkp ;
    (chunkp+1)->i = hash->i ;
    return( chunkp ) ;
}

/* MAKE_DEF defines the object CHUNK of name STRING to be in the tabulation 
   table INT. */

static gen_chunk *
make_def( Int, String, Chunk )
int Int ;
gen_chunk *String, *Chunk ;
{
    int domain ;
    char *id ;

    sscanf( String->s, "%d", &domain ) ;
    id = strchr( String->s, HASH_SEPAR )+1 ;

    return( enter_tabulated_def( Int, domain, id, Chunk, allow_forward_ref )) ;
}

/* MAKE_REF references the object of hash name STRING in the tabulation table
   INT. Forward references are dealt with here. */

static gen_chunk * make_ref(int Int, gen_chunk *String)
{
    gen_chunk *hash ;
    gen_chunk *cp ;
    int domain;

    if( Gen_tabulated_[ Int ] == (gen_chunk *)NULL ) {
	user( "read: Unloaded tabulated domain %s\n", Domains[ Int ].name ) ;
    }

    sscanf(String->s, "%d", &domain);
    domain = gen_type_translation_old_to_actual(domain);

    if( (hash=(gen_chunk *)gen_get_tabulated_name_direct(String->s))
	== (gen_chunk *)HASH_UNDEFINED_VALUE ) {
	if( allow_forward_ref ) {
	    hash = (gen_chunk *)alloc( sizeof( gen_chunk )) ;
	    hash->i = -gen_find_free_tabulated( &Domains[ domain ] ) ;

	    gen_put_tabulated_name_direct(String->s, (char *)hash) ;

	    if((Gen_tabulated_[ Int ]+abs( hash->i ))->p != 
	       gen_chunk_undefined) {
	        fatal("make_ref: trying to re-allocate for %s\n", String->s) ;
	    }
	    (Gen_tabulated_[ Int ]+abs( hash->i ))->p = 
		(gen_chunk *)alloc( gen_size( Domains+domain )* sizeof( gen_chunk )) ;
        }
	else {
	    user("make_ref: Forward references to %s prohibited\n",
		 String->s) ;
        }
    }
    cp = (Gen_tabulated_[ Int ]+abs( hash->i ))->p ;
    (cp+1)->i = abs( hash->i ) ;
    return( cp ) ;
}
