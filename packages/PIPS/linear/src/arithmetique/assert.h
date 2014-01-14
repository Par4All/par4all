/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* "abort" version of "/usr/include/assert.h", and assert with a message.
 * breakpoint on abort() to catch an issue.
 * put here so as to mask the "/usr/include" version.
 *
 * You need an include of <stdio.h> and <stdlib.h> to use it.
 */

#ifdef __GNUC__
#define _linear_assert_message                                          \
  "[%s] (%s:%d) assertion failed\n", __FUNCTION__, __FILE__, __LINE__
#else
#define _linear_assert_message                          \
  "Assertion failed (%s:%d)\n", __FILE__, __LINE__
#endif

#undef assert
#ifdef NDEBUG
#define assert(ex)
#define linear_assert(msg, ex)
#else
#define assert(ex) {                                            \
    if (!(ex)) {                                                \
      (void) fprintf(stderr, _linear_assert_message);           \
      (void) abort();                                           \
    }                                                           \
  }
#define linear_assert(msg, ex) {								\
    if (!(ex)) {                                                \
      (void) fprintf(stderr, _linear_assert_message);           \
      (void) fprintf(stderr, "\n %s not verified\n\n", msg);    \
      (void) abort();                                           \
    }                                                           \
  }
#endif /* NDEBUG */
