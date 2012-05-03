// $Id$
// http://research.microsoft.com/en-us/um/people/sumitg/benchmarks/pa.html.

#include <stdlib.h>
#define assume(c) {if (!(c)) exit(0);}
#define assert(c) {if (!(c)) exit(1);}

void twoloop(int x, int y, int m, int n) {
	assume(n >= 0);
	y = 0;
	while (y < n) {
		y++;
	}
	x = 0;
	while (x < n) {
		x++;
	}
	assert(x == y);
}

/* Predicate Set S:
 *
 * 	y <= 0, y > 0, y < 0, y >= 0,
 * 	x <= 0, x > 0, x < 0, x >= 0,
 * 	x - n <= 0, x - n > 0, x - n < 0, x - n >= 0,
 * 	y - n <= 0, y - n > 0, y - n < 0, y - n >= 0,
 * 	x - y <= 0, x - y > 0, x - y < 0, x - y >= 0
 */
