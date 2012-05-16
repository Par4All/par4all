// $Id$
// http://research.microsoft.com/en-us/um/people/sumitg/benchmarks/pa.html.

#include <stdlib.h>
#define assume(c) {if (!(c)) exit(0);}
#define assert(c) {if (!(c)) exit(1);}

void nested_bug(int x, int y, int m, int t) {

	assume(y == 0 && m >= 0 && t == 0);

	while (y < m) {
		y++;
		t = 0;
		while (t < y)
			t++;
	}

	assert(t == m);
}


/*
 * Predicate Set S:
 * 	y <= 0, y > 0, y < 0, y >= 0,
 * 	y - t <= 0, y - t > 0, y - t < 0, y - t >= 0,
 * 	t - m <= 0, t - m > 0, t - m < 0, t - m >= 0,
 * 	y - m <= 0, y - m > 0, y - m < 0, y - m >= 0
 */

