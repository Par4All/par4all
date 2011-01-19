// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385
// example 1

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (i <= n)

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

#define G1 (t != k && t < n)
#define G1a (t != k && t < n - 1)
#define G1b (t != k && t == n - 1)
#define G1c (t < k - 1 && t < n)
#define G1d (t == k - 1 && t < n)
#define U1 {t++; i++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}
#define T1c {trans(G1c, U1);}
#define T1d {trans(G1d, U1);}

#define G2 (t != k && t >= n)
#define G2a (G2 && 0 < k)
#define G2b (G2 && 0 == k)
#define U2 {t = 0; i++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(k < t && t < n);}
#define S2 {assert(t >= n);}
#define S3 {assert(t < k);}
#define S4 {assert(t == k);}

void run(void) {
	
	int k, n, t, i;
	k = rand();
	n = rand();
	t = k + 1;
	i = 0;
	
	if (0 <= k && k < n) {

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
		
		if (t < n) {
			S1;
			while (flip()) {
				T1a; S1;
			}
			T1b;
		}
		
		S2;
		if (flip()) {
			T2b; S4;
		}
		else {
			T2a; S3;
			while (flip()) {
				T1c; S3;
			}
			T1d; S4;
		}
		
#endif

	}
	
}

int main(void) {
	run();
	return 0;
}

