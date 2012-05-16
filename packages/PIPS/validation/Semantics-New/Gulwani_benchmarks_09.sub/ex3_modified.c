// $Id: ex3.c 5617 2012-05-02 14:48:53Z maisonneuve $
// http://research.microsoft.com/en-us/um/people/sumitg/benchmarks/pa.html.

#include <stdlib.h>
#define assume(c) {if (!(c)) exit(0);}
#define assert(c) {if (!(c)) exit(1);}

void ex3_modified(int x, int y, int n, int m) {
	assume(x == 0 && y == m && n >= m && m >= 0);
	while (x < n) {
		while (x<n && x < m)
			x++;
		while(x<n && x>=m) {
			x++;
			y++;
		}
	}
	assert(y == n);
}

/*
 * Predicate Set S:
 *	x - m <= 0, x - m > 0, x - m < 0, x - m >= 0,
 *	x - n <= 0, x - n > 0, x - n < 0, x - n >= 0,
 *	y - m <= 0, y - m > 0, y - m < 0, y - m >= 0,
 *	y - n <= 0, y - n > 0, y - n < 0, y - n >= 0,
 *	x - y <= 0, x - y > 0, x - y < 0, x - y >= 0,
 */
