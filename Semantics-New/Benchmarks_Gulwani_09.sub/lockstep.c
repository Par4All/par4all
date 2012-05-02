// $Id$
// http://research.microsoft.com/en-us/um/people/sumitg/benchmarks/pa.html.

#include <stdlib.h>
#define assume(c) {if (!(c)) exit(0);}
#define assert(c) {if (!(c)) exit(1);}

void lockstep(int x, int y, int n) {
	x = 0;
	y = 0;
	assume(n >= 0);
	while (x < n) {
		x++;
		y++;
	}
	assert(y == n);
}

/* 
 * Predicate Set S:
 *
 *	x - y >= 0,x - y <= 0,
 *	x - n <= 0,x - n >= 0,
 *	y - n <= 0
 */
