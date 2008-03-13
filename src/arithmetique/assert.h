/* Version "abort" de l'assert de /usr/include/assert.h 
 * Il est installe dans Linear de maniere a masquer /usr/include/assert.h
 *
 * You need an include of <stdio.h> and <stdlib.h> to use it.
 */

# ifndef NDEBUG
# define _assert(ex)	{if (!(ex)){(void)fprintf(stderr,"Assertion failed: file \"%s\", line %d\n", __FILE__, __LINE__);(void) abort();}}
# define assert(ex)	_assert(ex)
# else
# define _assert(ex)
# define assert(ex)
# endif
