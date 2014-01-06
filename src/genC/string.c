/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
  convenient functions to deal with strings.
  moved from pips by FC, 16/09/1998.

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
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <memory.h>

#include <stdarg.h>
#include "genC.h"

/* Like strdup() but copy at most n characters.
   The return string *is not* null terminated. if the original string is a
   least n-byte long. */
string gen_strndup(
    string s, /* la chaine a copier */
    size_t n /* le nombre de caracteres a copier */)
{
  size_t i;

  string r = (string) malloc(n);
  message_assert("allocated", r);

  /* recopie */
  for (i = 0; i < n && s[i] != '\0'; i += 1 )
    r[i] = s[i];

  /* padding */
  while (i < n) {
    r[i] = '\0';
    i += 1;
  }

  return r;
}

/* Like strdup() but copy at most n characters.
   The return string is null terminated. */
string gen_strndup0(
    string s, /* la chaine a copier */
    size_t n /* le nombre de caracteres a copier */)
{
  register string r;
  register size_t i;

  r = (string) malloc(n+1);
  message_assert("allocated", r);

  /* recopie */
  for (i = 0; i < n && s[i] != '\0'; i += 1 )
    r[i] = s[i];

  /* padding */
  while (i < n+1) {
    r[i] = '\0';
    i += 1;
  }

  return r;
}

/* CONCATENATE() *********** Last argument must be NULL *********/

#define BUFFER_SIZE_INCREMENT 128
#ifndef MAX
#define MAX(x,y) (((x)>(y))?(x):(y))
#endif

static string buffer = (string) NULL;
static size_t buffer_size = 0;
static size_t current = 0;

/* ITOA (Integer TO Ascii) yields a string for a given Integer.

   Moved in this global place from build.c since it used in many place in
   PIPS.*/
char * itoa(int i) {
  static char buf[ 20 ] ;
  sprintf( &buf[0], "%d", i ) ;
  return buf;
}

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

/* If the string is undefined, just skip it. Well, I hope it will not hide
   some bugs by masking some deliberate string_undefined put to trigger a
   wrong assertion is some higher parts of the code... */
string append_to_the_buffer(const char* s /* what to append to the buffer */)
{
  if (s != string_undefined)
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
  }

  return buffer;
}

string get_the_buffer(void)
{
    return buffer;
}

/* Return the concatenation of the given strings.
 *
 * CAUTION! concatenation is based on a static dynamically allocated buffer
 * which is shared from one call to another.
 *
 * Note that if a string is string_undefined, it is just skiped.
 *
 * FC.
 */
string concatenate(const char* next, ...)
{
  int count = 0;
  va_list args;
  char * initial_buffer = buffer;

  if (next && next!=initial_buffer)
    init_the_buffer();
  /* else first argument is the buffer itself... */

  /* now gets the strings and concatenates them
   */
  va_start(args, next);
  while (next)
  {
    count++;
    if (next!=initial_buffer) /* skipping first argument if is buffer */
      (void) append_to_the_buffer(next);
    next = va_arg(args, string);
    message_assert("reuse concatenate result only as the first argument",
		   !next || next!=initial_buffer);
    /* should stop after some count? */
  }
  va_end(args);

  /* returns the static '\0' terminated buffer.
   */
  return buffer;
}

string strupper(string s1, const char* s2)
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

string strlower(string s1, const char* s2)
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
  return b? "true": "false";
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


/* Find if a string s end with a suffix.

   If yes, return a pointer to the suffix in s, if not, return NULL. */
string find_suffix(string s, string suffix) {
  size_t l = strlen(s);
  size_t l_suffix = strlen(suffix);

  if (l < l_suffix)
    /* No way if the suffix is longer than the string! */
    return NULL;

  string suffix_position = s + l - l_suffix;
  if (memcmp(suffix_position, suffix, l_suffix) == 0)
    /* Get it! */
    return suffix_position;

  /* Not found: */
  return NULL;
}

///@return a string whithout the last "\n" char
///@param s, the string to process
///@param flg, if true do a copy of the string
string chop_newline (string s, bool flg)
{
  if ((s == string_undefined) || (s == NULL)) return s;
  string r = s;
  if (flg == true) r = strdup(s);
  int l = strlen(s);
  if (l > 0) {
    if (*(r + l - 1) == '\n') *(r + l - 1) = '\0';
  }
  return r;
}


///@return a copy of the string whithout the last "\n" char
///@param s, the string to process
string remove_newline_of_string (string s)
{
  return chop_newline (s, true) ;
}

/* @return array of string
 * @param s string  considered
 * @pram d delimiter
 */
list strsplit(const char *s,const char *d)
{
    string buffer=strdup(s);
    list split = NIL;
    for(string tmp = strtok(buffer,d); tmp;tmp=strtok(NULL,d))
        split=CONS(STRING,strdup(tmp),split);
    free(buffer);
    return gen_nreverse(split);
}

/**
 * Callback for sorting string with qsort
 * @return see man strcmp
 */
int gen_qsort_string_cmp(const void * s1, const void *s2) {
  return strcmp(*(char **)s1, *(char **)s2);
}

/**
 * @brief Prepend the prefix to the string. Free the destination string
 * if needed
 * @param dst, the string to be modified
 * @param prefix, the prefix to be prepended
 */
void str_prepend (string* dst, string prefix) {
  if ((prefix == NULL) || prefix == string_undefined || strlen (prefix) == 0)
	return;
  string old = * dst;
  *dst = strdup (concatenate (prefix, *dst,NULL));
  free (old);
}

/**
 * @brief Append the suffix to the string. Free the destination string
 * if needed
 * @param dst, the string to be modified
 * @param suffix, the suffix to be appended
 */
void str_append (string* dst, string suffix) {
  if ((suffix == NULL) || suffix == string_undefined || strlen (suffix) == 0)
	return;
  string old = * dst;
  *dst = strdup (concatenate (*dst, suffix, NULL));
  free (old);
}
