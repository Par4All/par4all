/* package arithmetic
 *
 * $RCSfile: io.c,v $ (version $Revision$)
 * $Date: 1996/07/13 15:55:25 $, 
 */
#include <stdio.h>
#include <strings.h>

#include "arithmetique.h"

/* this seems a reasonnable upperbound
 */
#define BUFFER_SIZE 50

char * Value_to_string(Value v)
{
    static char buf[BUFFER_SIZE];
    return sprintf(VALUE_FMT "\0", buf, v);
}

void fprint_Value(FILE *f, Value v)
{
    (void) fprintf(f, VALUE_FMT, v);
}

char * sprint_Value(char *s, Value v)
{
    return sprintf(s, VALUE_FMT, v);
}

void fscan_Value(FILE *f, Value *pv)
{
    (void) fscanf(f, VALUE_FMT, pv);
}

/* end of $RCSfile: io.c,v $
 */
