/*

CONCATENATE concatenates a variable list of strings in a static string
(which is returned).

STRNDUP copie les N premiers caracteres de la chaine S dans une zone
allouee dynamiquement, puis retourne un pointeur sur cette zone. si la
longueur de S est superieure ou egale a N, aucun caratere null n'est
ajoute au resultat. sinon, la chaine resultat est padde avec des
caracteres null.


STRNDUP0 copie les N premiers caracteres de la chaine S dans une zone
allouee dynamiquement, puis retourne un pointeur sur cette zone.
l'allocation est systematiquement faite avec N+1 caracteres de telle
sorte qu'un caractere null soit ajoute a la chaine resultat, meme si la
longueur de la chaine S est superieure ou egale a N. dans le cas
contraire, la chaine resultat est padde avec des caracteres null.

*/

#include <stdlib.h>
#include <stdio.h>
/*
extern int fprintf();
extern char toupper(char c);
*/
#include <stdarg.h>
#include "string.h"
#include "genC.h"
#include <ctype.h>

#include "misc.h"

string strndup(n, s)
register int n; /* le nombre de caracteres a copier */
register string s; /* la chaine a copier */
{
 	register string r;
	register int i;

	/* allocation */
	if ((r = (string) malloc((unsigned) n)) == NULL) {
		fprintf(stderr, "strndup: out of memory\n");
		exit(1);
	}

	/* recopie */
	for (i = 0; i < n && s[i] != NULL; i += 1 )
		r[i] = s[i];

	/* padding */
	while (i < n) {
		r[i] = '\0';
		i += 1;
	}

	return(r);
}

string strndup0(n, s)
register int n; /* le nombre de caracteres a copier */
register string s; /* la chaine a copier */
{
 	register string r;
	register int i;

	/* allocation */
	if ((r = (string) malloc((unsigned) n+1)) == NULL) {
		fprintf(stderr, "strndup0: out of memory\n");
		exit(1);
	}

	/* recopie */
	for (i = 0; i < n && s[i] != NULL; i += 1 )
		r[i] = s[i];

	/* padding */
	while (i < n+1) {
		r[i] = '\0';
		i += 1;
	}

	return(r);
}

/* CONCATENATE() *********** Last argument must be NULL *********/
/*VARARGS0*/
string
concatenate(char * first_string, ...)
{
#define CONCATENATE_BUFFER_SIZE 10240
   va_list args ;
   static char result[ CONCATENATE_BUFFER_SIZE ] ;
   char *p ;
   char * next_string ;
   char * current = &result[0];
   char * end = &result[ CONCATENATE_BUFFER_SIZE - 1 ];

   va_start( args, first_string ) ;
   result[ 0 ] = '\0' ;

   p = first_string;
   
   do {
      /* strcat( result, p ) ; */
      char * pc;

      next_string = va_arg( args, char * );
      for(pc = p; *pc; pc++, current++)
         if( current < end )
            *current = *pc;
         else
            /* a larger buffer could be malloc'ed... */
            pips_error("concatenate", "buffer overflow\n");
      p = next_string;
   }
   while( next_string != NULL );

   *current = '\0';
   va_end( args ) ;
   return( result ) ;
}

char *strupper(s1, s2)
char *s1, *s2;
{
    char *r = s1;

    while (*s2) {
	*s1 = (islower(*s2)) ? toupper(*s2) : *s2;
	s1++;
	s2++;
    }

    *s1 = '\0';
	
    return(r);
}

string bool_to_string(b)
bool b;
{
    static string t = "TRUE";
    static string f = "FALSE";

    return b?t:f;
}
