/* $Id$
   $Log: c_syntax-local.h,v $
   Revision 1.2  2003/08/04 14:21:10  nguyen
   Preliminary version of the C parser

   Revision 1.1  2003/06/24 07:25:13  nguyen
   Initial revision

*/

extern FILE * c_in; /* the file read in by the c_lexer */

/* The following declarations are used to avoid warning with implicit declarations,
   although include <stdio.h>, include <string.h> are already added, I do not know why :-)*/
extern int fileno(FILE *stream);
extern char *strdup(const char *s1); 
