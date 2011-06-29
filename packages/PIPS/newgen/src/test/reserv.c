

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "lieu.h"
#include "newres.h"
#include "spec.h"

/* declarations des fonctions */
personne lire_passager() ;
personne lire_conducteur() ;
lieu lire_lieu() ;
date lire_date() ;
reservation lire_reservation() ;

int secu_id = 0 ;
identification secu ;

struct mylogin {
	char name ;
	char user_id ;
};

typedef char *login ;

string mk_string(s)
string s;
{
    if (s != NULL)
	    return(strcpy((string) malloc(strlen(s)+1), s));
    else
	    abort() ;
/*	    return(NULL);*/
}

#define LGBUF 256
string ld_string(fic)
FILE * fic;
{
    static char buffer[LGBUF+1];
    register int i;

    for (i = 0; i < LGBUF && (buffer[i] = getc(fic)) != '\n'; i++) ;

    if (i == LGBUF) {
	fprintf(stderr, 
		"ld_string: buffer trop petit, recompiler en augmentant LGBUF");
	exit(2);
    }
    buffer[i] = '\0';

    return(mk_string((string) buffer));
}

FILE * ouvrir(f, m)
string f;
string m;
{
    FILE * fd;

    if ((fd = fopen(f, m)) == NULL) {
	fprintf(stderr, 
		"ouvrir: ouverture impossible (fichier: %s  -  mode: %s)\n", f, m);
	exit(1);
    }
    return(fd);
}

int ld_int(fic)
FILE * fic;
{
    int i;
    fscanf(fic,"%d",&i);
    while (getc(fic) != '\n');
    return(i);
}

char ld_char(fic)
FILE * fic;
{
    char c;
    c = getc(fic);
    while (getc(fic) != '\n');
    return(c);
}

reservation lire_reservation()
{
    reservation r;
    int i;
    personne p = (personne) lire_conducteur() ;
    date d = (date) lire_date() ;
    lieu l = (lieu) lire_lieu() ;

    r = make_reservation( personne_undefined, d, l, 0, NULL, date_undefined );
    reservation_conducteur( r ) = p ;
    reservation_passager(r)[0] = reservation_conducteur(r);

    /*	reservation_conducteur(r) = 
	PERSONNE(reservation_passager(r)[0]) = reservation_conducteur(r);
	reservation_date(r) = 
	reservation_destination(r) = 
	*/
    fprintf(stderr, "Combien de passager transporterez-vous ? ");
    if ((reservation_nbpassager(r) = ld_int(stdin)) < 0)
	    reservation_nbpassager(r) = 0;
    else if (reservation_nbpassager(r) > 3) {
	fprintf(stderr, "Le nombre de passager est limite a 3\n");
	reservation_nbpassager(r) = 3;
    }

    for (i = 1; i <= reservation_nbpassager(r); i++) {
	fprintf(stderr, "Nom du passager %d ? ", i);
	reservation_passager(r)[i] = lire_passager();
    }
    fprintf(stderr, "Cette reservation est-elle definitive (o/n) ? ");
    reservation_a_confirmer(r) = (ld_char(stdin) == 'o') ? d : d ;
    gen_write( stderr, r );
    return(r);
}

personne lire_conducteur()
{
    personne p;
    struct mylogin *myl = (struct mylogin *)malloc( sizeof( struct mylogin )) ;
    set logins = set_make( set_pointer ) ;

    myl->name = *mk_string((string) getenv("LOGNAME"));
    myl->user_id = '9' ;
    fprintf(stderr, "Votre nom ? ");
    set_add_element( logins, logins, (login)myl ) ;
    p = make_personne( ld_string(stdin), logins ) ;
    extend_identification(secu, p, secu_id++) ;
    return(p);
}

personne lire_passager()
{
    personne p;
    struct mylogin *myl = (struct mylogin *)malloc( sizeof( struct mylogin )) ;
    set logins = set_make( set_pointer ) ;

    myl->name = 'p' ;
    myl->user_id = '8' ;
    set_add_element( logins, logins, (login)myl ) ;
    p = make_personne( ld_string(stdin), logins ) ;
    extend_identification(secu, p, secu_id++) ;
    return(p);
}

date lire_date()
{
    date d; periode p;
    char c;
    string date_id ;

    p = make_periode( is_periode_MATIN, UU );

    date_id = malloc( 100 ) ;
    sprintf( date_id, "_d%d", (int) (1000*( 0xFFFF&rand())) ) ;
    hash_warn_on_redefinition() ;
    d = make_date( date_id, 0, 0, 0, p ) ;
    d = make_date( date_id, 0, 0, 0, p ) ;
    fprintf(stderr, "Date (jj mm aa) ? ");
    fscanf(stdin, "%d%d%d\n", 
	   &(date_jour(d)), &(date_mois(d)), &(date_annee(d)));

    do {
	fprintf(stderr, "M(atin) A(pres-midi) J(ournee) ? ");
	c = ld_char(stdin);
	switch (c) {
	case 'M': periode_tag(p) = is_periode_MATIN; break;
	case 'A': periode_tag(p) = is_periode_APRESMIDI; break;
	case 'J': periode_tag(p) = is_periode_JOURNEE; break;
	}
    } while (c != 'M' && c != 'A' && c != 'J');
    date_periode(d) = p;
    return(d);
}

lieu lire_lieu()
{
    lieu l;
    char c;
    set s ;

    l = make_lieu( is_lieu_connu, connu_undefined ) ;

    do {
	fprintf(stderr,
		"Destination P(aris) S(ophia) R(ocquencourt) A(utre) ? ");
	c = ld_char(stdin);
	if (c == 'P' || c == 'S' || c == 'R') {
	    connu lc = make_connu(is_connu_PARIS, UU);
	    lieu_tag(l) = is_lieu_connu;
	    lieu_connu(l) = lc;
	    switch (c) {
	    case 'P': connu_tag(lc) = is_connu_PARIS; break;
	    case 'S': connu_tag(lc) = is_connu_SOPHIA; break;
	    case 'R': connu_tag(lc) = is_connu_ROCQUENCOURT; break;
	    }
	}
	else if (c == 'A') {
	    lieu_tag(l) = is_lieu_autre;
	    if( !lieu_autre_p( l )) {
		fprintf(stderr, "Assert failed" ) ;
		exit( 1 ) ;
	    }
	    fprintf(stderr, "Nom de cette nouvelle destination ? ");
	    s = set_singleton( set_pointer, ld_string(stdin) ) ;
	    set_add_element( s, s, strdup("foo") ) ;
	    lieu_autre(l) = s ;
	}
    } while (c != 'P' && c != 'S' && c != 'R' && c != 'A');
    return(l);
}

void
login_write( fd, l )
FILE * fd;
login l ;
{
    struct mylogin *myl = (struct mylogin *)l ;
    fprintf( fd, "%c%c", myl->name, myl->user_id ) ;
}

login
login_read( f, read )
FILE *f ;
int (*read)();
{
    struct mylogin *myl = (struct mylogin *)malloc( sizeof( struct mylogin ));

    myl->name = read() ;
    myl->user_id = read() ;
    fprintf( stderr, "Read %c%c\n", myl->name, myl->user_id ) ;
    return( (login)myl ) ;
}

void
login_free( l )
login l ;
{
    /* FI: free( (struct mylogin *)l ); */
    free( (char *)l );
}

login
login_copy( l )
login l ;
{
    return( l ) ;
}

void
print_date( d )
date d ;
{
    fprintf( stderr, "date %s, offset %d\n", date_ident( d ) ,
	     ((gen_chunk *)d+1)->i ) ;
}

int main(void)
{
    cons * l;
    indisponibilite i, ii;
    identification s1;
    reservation r;
    char ch;
    FILE * fd;

    gen_read_spec( ALL_SPECS ) ;
    gen_init_external(LOGIN_NEWGEN_EXTERNAL, 
		      login_read, login_write, login_free, login_copy, 
		      gen_false ) ;
    gen_debug = 0 ;
/*    gen_debug = GEN_DBG_TRAV_LEAF|GEN_DBG_TRAV_OBJECT ;*/
    i = make_indisponibilite( (cons *)NULL, identification_undefined );
    secu = make_identification() ;

    /* Only works for even number of reservations -- to test nested CONS */

    do {
	reservation r1, r2 = reservation_undefined;

	fprintf(stderr, "Voulez-vous faire une nouvelle reservation (o/n) ? ");
	if ((ch = ld_char(stdin)) == 'o') {
	    r1 = (reservation) lire_reservation();
	} else break ;
	fprintf(stderr, "Voulez-vous faire une nouvelle reservation (o/n) ? ");
	if ((ch = ld_char(stdin)) == 'o') {
	    r2 = (reservation) lire_reservation();
	} 
	indisponibilite_reservation(i) = 
		CONS( RESERVATION, r2, 
		     CONS( RESERVATION, r1, indisponibilite_reservation(i)));
    } while (ch == 'o');


    fprintf( stderr, "End input\n") ;
    fd = ouvrir( "sortie4", "w" ) ;
    write_identification( stderr, secu ) ;
    write_identification( fd, secu ) ;
    fflush( fd ) ;
    fclose( fd ) ;
    fprintf( stderr, "End writing secu\n") ;
    fd = ouvrir( "sortie4", "r" ) ;
    s1 = read_identification( fd ) ;
    write_identification( stderr, s1 ) ;
    fprintf( stderr, "End secu\n") ;

    indisponibilite_reservation(i) = 
	    gen_nreverse( indisponibilite_reservation(i)) ;
    fprintf( stderr, "End input\n") ;

    for (l = indisponibilite_reservation(i); 
	 !ENDP(l); 
	 l = CDR(l)) 
    {
	int j ;
	fprintf(stderr, "Reservations restantes %d\n", gen_length(l)) ;
	r = RESERVATION(gen_nth(0,l)) ;

	j = apply_identification(secu, reservation_conducteur(r)) ;
	update_identification(secu, reservation_conducteur(r), 100+j ) ;
	j = apply_identification(secu, reservation_conducteur(r)) ;

	fprintf(stderr, "%s (%d) a reserve la voiture le %d/%d/%d\n pour %s,",
		personne_nom(reservation_conducteur(r)),
		j,
		date_jour(reservation_date(r)),
		date_mois(reservation_date(r)),
		date_annee(reservation_date(r)),
		personne_nom(reservation_passager(r)[1])) ;
	SET_MAP( log, {fprintf(stderr, "%c\n",
			       ((struct mylogin *)log)->user_id ) ;},
		 personne_logins(reservation_passager(r)[1])) ;
    }
    fprintf( stderr, "End report\n" ) ;
    fd = ouvrir("sortie", "w");
    gen_debug = GEN_DBG_CHECK ;
    /*   gen_debug = GEN_DBG_TRAV ; */
    gen_write_tabulated( fd, date_domain ) ;
    fprintf( stderr, "End writing dates\n" ) ;
    fprintf( stderr, "i Consistent = %d\n", gen_consistent_p( i )) ;
    fprintf( stderr, "i Defined = %d\n", gen_defined_p( i)) ;
    gen_write(fd, i);
    fflush( fd ) ;
    fclose( fd ) ;
    fd = ouvrir( "sortie3", "w" );
    fprintf(stderr, "i Consistent before write = %d\n", 
	    gen_consistent_p( i )) ;
    gen_write(fd, i);
    fprintf(stderr, "i Consistent after write = %d\n", 
	    gen_consistent_p( i )) ;
    fflush( fd ) ;
    fclose( fd ) ;
    fd = ouvrir( "sortie3", "r" ) ;
    fprintf( stderr, "Read Consistent after read = %d\n",
	    gen_consistent_p( gen_read( fd ) )) ;
    fprintf( stderr, "i Consistent after read = %d\n",
	    gen_consistent_p( i )) ;
    fclose( fd ) ;

    gen_mapc_tabulated( print_date, date_domain ) ;
    fprintf(stderr, "Mapc tabulated\n" ) ;
    gen_free( i ) ;
    fprintf( stderr, "I freed\n" ) ;
    gen_free_tabulated( date_domain ) ;
    fprintf( stderr, "Dates freed\n" ) ;
    fd = ouvrir( "sortie", "r" ) ;
    gen_read_and_check_tabulated( fd, true ) ;
    fprintf( stderr, "Table read\n" ) ;
    gen_mapc_tabulated( print_date, date_domain ) ;
    i = (indisponibilite)gen_read( fd ) ;
    fprintf( stderr, "End reading\n" ) ;
/*    gen_debug = GEN_DBG_TRAV ; */
    gen_write( stderr, i ) ;
    gen_debug = GEN_DBG_CHECK ;

    for (l = indisponibilite_reservation(i); l != (cons *)NULL; l = CDR(l)) {
	r = RESERVATION(CAR(l)) ;
	fprintf(stderr, "%s a reserve la voiture le %d/%d/%d\n pour %s,",
		personne_nom(reservation_conducteur(r)),
		date_jour(reservation_date(r)),
		date_mois(reservation_date(r)),
		date_annee(reservation_date(r)),
		personne_nom(reservation_passager(r)[1]));
	SET_MAP( log, {fprintf(stderr, "%c\n",
			       ((struct mylogin *)log)->user_id ) ;},
		 personne_logins(reservation_passager(r)[1])) ;
    }
    fd = ouvrir( "sortie2", "w" ) ;
    gen_write_tabulated( fd, date_domain ) ;

#warning "TABULATED_MAP is not implemented anymore..."
    /*
    TABULATED_MAP( d, {
	fprintf( stderr, "%s\n", date_ident( d ));
      },
      date_domain);
    */

    gen_debug = GEN_DBG_CHECK ;
/*    gen_debug = GEN_DBG_TRAV ;*/
    fprintf(stderr, "%d reservations\n",
	    gen_length(indisponibilite_reservation(i)));
/*    gen_debug = GEN_DBG_TRAV ; */
    ii = copy_indisponibilite( i ) ;
    gen_debug = GEN_DBG_CHECK ;
    gen_write( fd, i ) ;
    fprintf( stderr, "Writing copy\n" ) ;
    gen_write( fd, ii ) ;
    fprintf( stderr, "End writing\n" ) ;
    fprintf( stderr, "Check sharing\n" ) ;
    gen_debug = GEN_DBG_TRAV ;
    l = indisponibilite_reservation( i ) ;
    l = CONS( RESERVATION, RESERVATION(CAR( l )),
	      CONS( RESERVATION, RESERVATION(CAR( l )), NIL)) ;
    gen_write( stderr, make_indisponibilite( l, secu )) ;
    RESERVATION_(CAR(l)) = reservation_undefined ;
    fprintf(stderr, "i Defined = %d\n", 
	    gen_defined_p( make_indisponibilite( l, secu ))) ;
    gen_free( make_indisponibilite( l, secu )) ;
    fflush( fd ) ;

    fprintf( stderr, "Test 1" ) ;
    if( gen_sharing_p( i, i ) != 1 ) {
      abort() ;
    }
    fprintf( stderr, "Test 2" ) ;
    if(gen_sharing_p(i,RESERVATION(CAR(indisponibilite_reservation(i))))!=1){ 
      abort() ;
    }
    fprintf( stderr, "Test 3" ) ;
    if( gen_sharing_p( i, ii ) != 0 ) {
      abort() ;
    }
    fprintf( stderr, "Test 4" ) ;
    if(gen_sharing_p(i,RESERVATION(CAR(indisponibilite_reservation(ii))))!=0){ 
      abort() ;
    }
    fprintf( stderr, "Test 5" ) ;
    indisponibilite_reservation(i) = CDR(indisponibilite_reservation(ii)) ;
    if( gen_sharing_p( i, ii ) != 1 ) {
      abort() ;
    }

    gen_free( i ) ;
    
    MAPL(ls, {
	fprintf(stderr, "%d", INT( gen_nth( 0, ls )));
    }, CONS( INT, 1, CONS( INT, 2, NIL ))) ;

    fprintf(stderr,"\n");
	return 0;
}
