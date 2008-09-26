/* $Id$ */

#ifndef BOOLEAN_INCLUDED
#define BOOLEAN_INCLUDED

/* put here because should be included before any other header in linear */
#ifdef CPROTO_ATTRIBUTE_FIX
#define __attribute__(x) /* nope! */
#endif /* old cproto attribute fix for Ronan */

typedef enum { false, true } boolean;
#define	TRUE     true
#define	FALSE    false

#define boolean_string(b) ((b)? "TRUE" : "FALSE")

#endif /* BOOLEAN_INCLUDED */
