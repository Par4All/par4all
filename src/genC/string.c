/*
  convenient functions to deal with strings.
  moved from pips by FC, 16/09/1998.

  $Id$
  
*/

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
#include <ctype.h>
#include <memory.h>

#include <stdarg.h>
#include "genC.h"

string gen_strndup(
    string s, /* la chaine a copier */
    size_t n /* le nombre de caracteres a copier */)
{
 	register string r;
	register size_t i;

	/* allocation */
	if ((r = (string) malloc(n)) == NULL) {
		fprintf(stderr, "gen_strndup: out of memory\n");
		exit(1);
	}

	/* recopie */
	for (i = 0; i < n && s[i] != '\0'; i += 1 )
		r[i] = s[i];

	/* padding */
	while (i < n) {
		r[i] = '\0';
		i += 1;
	}

	return(r);
}

string gen_strndup0(
    string s, /* la chaine a copier */
    size_t n /* le nombre de caracteres a copier */)
{
 	register string r;
	register size_t i;

	/* allocation */
	if ((r = (string) malloc(n+1)) == NULL) {
		fprintf(stderr, "gen_strndup0: out of memory\n");
		exit(1);
	}

	/* recopie */
	for (i = 0; i < n && s[i] != '\0'; i += 1 )
		r[i] = s[i];

	/* padding */
	while (i < n+1) {
		r[i] = '\0';
		i += 1;
	}

	return(r);
}

/* CONCATENATE() *********** Last argument must be NULL *********/

#define BUFFER_SIZE_INCREMENT 128
#ifndef MAX
#define MAX(x,y) (((x)>(y))?(x):(y))
#endif

static string buffer = (string) NULL;
static size_t buffer_size = 0;
static size_t current = 0;

void init_the_buffer(void)
{
    /* initial allocation
     */
    if (buffer_size==0)
    {
		message_assert("NULL buffer", buffer==NULL);
		buffer_size = BUFFER_SIZE_INCREMENT;
		buffer = (string) malloc(buffer_size);
		message_assert("enough memory", buffer);
    }
    current = 0;
    buffer[0] = '\0';
}

string append_to_the_buffer(string s) /* what to append to the buffer */
{
    size_t len = strlen(s);

    /* reallocates if needed
     */
    if (current+len >= buffer_size)
    {
		buffer_size = MAX(current+len+1, 
						  buffer_size+BUFFER_SIZE_INCREMENT);
		buffer = realloc(buffer, buffer_size);
		message_assert("enough memory", buffer);
    }

    (void) memcpy(&buffer[current], s, len);
    current += len;
    buffer[current] = '\0' ;

    return buffer;
}

string get_the_buffer(void)
{
    return buffer;
}

/* concatenation is based on a static dynamic buffer
 * which is shared from one call to another. beurk.
 *
 * FC.
 */
string concatenate(string next, ...)
{
    int count = 0;
    va_list args;
    
    init_the_buffer();

    /* now gets the strings and concatenates them
     */
    va_start(args, next);
    while (next)
    {
		count++;
		(void) append_to_the_buffer(next);
		next = va_arg(args, string);

		/* should stop after some count. */
    }
    va_end(args);

    /* returns the static null terminated buffer
     */
    return buffer;
}

string strupper(string s1, string s2)
{
    char *r = s1;

    while (*s2) {
		*s1 = (islower((int)*s2)) ? toupper(*s2) : *s2;
		s1++;
		s2++;
    }

    *s1 = '\0';
	
    return r;
}

string strlower(string s1, string s2)
{
    char *r = s1;

    while (*s2) {
		*s1 = (isupper((int)*s2)) ? tolower(*s2) : *s2;
		s1++;
		s2++;
    }

    *s1 = '\0';
	
    return r;
}

string bool_to_string(bool b)
{
    return b? "TRUE": "FALSE";
}

/* @return the english suffix for i.
 */
string nth_suffix(int i)
{
    string suffix = string_undefined;

    message_assert("Formal parameter i is greater or equal to 1", i >= 1);

    switch(i) {
    case 1: suffix = "st";
	break;
    case 2: suffix = "nd";
	break;
    case 3: suffix = "rd";
	break;
    default: suffix = "th";
	break;
    }
    return suffix;
}
