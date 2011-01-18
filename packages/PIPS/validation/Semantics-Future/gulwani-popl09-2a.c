// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity. POPL
// 2009: 127-139
// example 2a

// $Id$

#define USE_ASSERT 0
#define GOOD (i <= n - x0 + m - y0)

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}

#define G1 (x < n && y < m)
#define G1a (x < n && y < m - 1)
#define G1b (x < n && y == m - 1)
#define C1 {y++; i++;}
#define T1 {trans(G1, C1);}
#define T1a {trans(G1a, C1);}
#define T1b {trans(G1b, C1);}

#define G2 (x < n && y >= m)
#define G2a (x < n - 1 && y >= m)
#define G2b (x == n - 1 && y >= m)
#define C2 {x++; i++;}
#define T2 {trans(G2, C2);}
#define T2a {trans(G2a, C2);}
#define T2b {trans(G2b, C2);}

#define S1 {assert(x < n && y < m);}
#define S2 {assert(x < n && y >= m);}
#define S3 {assert(x >= n && y >= m);}

void run(void) {
	
	int x, y, x0, y0, n, m, i;
	x0 = rand();
	y0 = rand();
	x = x0;
	y = y0;
	i = 0;
	n = rand();
	m = rand();

	if (x0 < n && y0 < m) {

		if (y < m) {
			S1;
			while (flip()) {
				T1a; S1;
			}
			T1b;
		}
		
		S2;
		while (flip()) {
			T2a; S2;
		}
		T2b;
		
		S3;
		
	}
}

int main(void) {
	run();
	return 0;
}

