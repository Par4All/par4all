/* package arithmetic
 *
 * $RCSfile: io.c,v $ (version $Revision$)
 * $Date: 1996/07/13 14:31:44 $, 
 */
#include <stdio.h>
#include <strings.h>

#include "assert.h"        /* abort version */
#include "arithmetique.h"

#define BUFFER_SIZE 21

#ifdef VALUE_IS_LONGLONG
#define BASE '9' /* last digit of the base, must be >= '1' */
#define STRING_ZERO "\00000000000000000000\0"
#define STRING_ONE  "\00000000000000000001\0"

/* adds two chars for BASE, and tells whether an increment is needed */
static char addchar(char a1, char a2, int *ret)
{
    char res = a1+a2-'0';
    if (res>BASE)
	res-=(BASE-'0'+1), *ret=1;
    else
	*ret = 0;
    return res;
}

/* a++ (for incrementing, starting from a and going to the left)
 */
static void addone(char *a)
{
    int ret=1;
    while(ret && *a)
	*a--=addchar(*a,'1',&ret);
    assert(!ret);
}

/* a+=b */
static void add(char *a, char *b)
{
    int i, ret;
    assert(a!=b);
    for (i=BUFFER_SIZE-1; i>=0; i--)
    {
	addchar(*(a+i), *(b+i), &ret);
	if (ret) addone(a+i);
    }
}

/* s*=2 */
static void timestwo(char *s)
{
    char buf[BUFFER_SIZE];
    (void) strncpy(buf, s, BUFFER_SIZE);
    add(s, buf);
}

/* returns a static pointer, thus must be strdup if necessary...
 * some format could be specified? "x" -> BASE is 'f', "d" -> '9'...
 */
char * Value_to_string(Value v)
{
   static char buf[BUFFER_SIZE]; 
   char pow[BUFFER_SIZE];
   char *pb;
   int negative = 0;
   Value power=VALUE_ONE;
   int iter;
   
   if (VALUE_NEG_P(v)) 
       v=-v, negative=1;

   (void) strncpy(buf, STRING_ZERO, BUFFER_SIZE);   /* buf = 0 */
   (void) strncpy(pow, STRING_ONE,  BUFFER_SIZE);   /* pow = 1 */

   /* compute the string value of v 
    */
   for (iter=(sizeof(Value)*8-1); iter>0; iter--)
   {
       if (power&v) add(buf, pow); /* buf += pow */
       timestwo(pow);              /* pow *= 2 */
       power<<=1;
   }
   
   /* remove leading zeros but the last
    */
   for (pb=buf+1; *pb=='0' && *(pb+1); pb++);

   /* add the sign of needed 
    */
   if (negative) pb--, *pb='-';

   return pb;
}
#else
char * Value_to_string(Value v)
{
    static char buf[BUFFER_SIZE];
    return sprintf("%ld\0", buf, v);
}
#endif

/* end of $RCSfile: io.c,v $
 */
