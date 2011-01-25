// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity. POPL
// 2009: 127-139
// example 2b

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 0
#define GOOD (i <= n - x0 + n - z0)

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

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#ifdef GOOD
#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}
#else
#ifdef BAD
#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}
#else
#define trans(g, c) {if (g) {c;} else freeze;}
#endif
#endif

// transition system

#define G1 (x < n && z > x)
#define G1a (x < n - 1 && z == x + 1)
#define G1b (x < n - 1 && z > x + 1)
#define G1c (x == n - 1 && z > x)
#define U1 {x++; i++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}
#define T1c {trans(G1c, U1);}

#define G2 (x < n && z <= x)
#define G2a (x < n && z < x)
#define G2b (x < n && z == x)
#define U2 {z++; i++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(x < n && z <= x);}
#define S2 {assert(x < n && z > x);}
#define S3 {assert(x >= n);}

void run(void) {
	
	int x, z, x0, z0, n, i;
	x0 = rand();
	z0 = rand();
	x = x0;
	z = z0;
	i = 0;
	n = rand();

	if (x0 < n && z0 < n) {

#if RESTRUCTURE == 0
		
		while (1) {
			if (flip()) {
				T1;
			}
			else {
				T2;
			}
		}
			
#else

		if (z <= x) {
			S1;
			while (flip()) {
				T2a; S1;
			}
			T2b;
		}
		
		S2;
		while (flip()) {
			if (flip()) {
				T1b; S2;
			}
			else {
				T1a; S1;
				while (flip()) {
					T2a; S1;
				}
				T2b; S2;
			}
		}
		
		T1c; S3;
		
#endif
	
	}

}

int main(void) {
	run();
	return 0;
}

