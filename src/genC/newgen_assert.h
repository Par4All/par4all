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
  abort version of assert.
  the message generates the function name if possible.
  message_assert prints a message before aborting
*/

#ifdef __GNUC__
#define _newgen_assert_message						\
  "[%s] (%s:%d) assertion failed\n", __FUNCTION__, __FILE__, __LINE__
#else
#define _newgen_assert_message				\
  "Assertion failed (%s:%d)\n", __FILE__, __LINE__
#endif

#undef assert
#ifdef NDEBUG
#define assert(ex)
#define message_assert(msg, ex)
#else
#define assert(ex) {						\
    if (!(ex)) { 						\
      (void) fprintf(stderr, _newgen_assert_message);		\
      (void) abort();						\
    }								\
  }
#define message_assert(msg, ex) {				\
    if (!(ex)) {						\
      (void) fprintf(stderr, _newgen_assert_message);		\
      (void) fprintf(stderr, "\n %s not verified\n\n", msg);	\
      (void) abort();						\
    }								\
  }
#endif /* NDEBUG */

/*  That is all
 */
