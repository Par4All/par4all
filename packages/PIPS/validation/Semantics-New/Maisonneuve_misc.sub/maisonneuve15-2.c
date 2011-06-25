// We consider the same transition system as maisonneuve15-1'. We try to
// restructure it by partitioning in three states, depending on which guards
// are satisfied:
//
// S1: G1, not G2
//
// S2: G2, not G1
//
// S3: G1 and G2
//
// (The state (not G1 and not G2) is not reachable.)
//
// Unfortunately, states S1 and S2 are not convex and PIPS fails to check the
// assertions related to these states, even with transformer lists. This
// restructuration also leads to define transitions whose guards are not
// convex.

// $Id$

#define USE_ASSERT 1
#define USE_CHECK 1
#define BAD (x == 1 && y == 5)

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

#define G1 (x >= 0 && x <= 4 && y >= 0 && y <= 4)
#define G1a (G1 && x <= 2 && y <= 2)
#define G1b (G1 && (x > 2 || y > 2))
#define G1c (G1 && x < 4 && y < 4)
#define G1d (G1 && x == 4 && y == 4)
#define U1 {x += 2; y += 2;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}
#define T1c {trans(G1c, U1);}
#define T1d {trans(G1d, U1);}

#define G2 (x >= 2 && x <= 6 && y >= 2 && y <= 6)
#define G2a (G2 && x >= 4 && y >= 4)
#define G2b (G2 && (x < 4 || y < 4))
#define G2c (G2 && x > 2 && y > 2)
#define G2d (G2 && x == 2 && y == 2)
#define U2 {x -= 2; y -= 2;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}
#define T2c {trans(G2c, U2);}
#define T2d {trans(G2d, U2);}

#define S1 {assert(G1 && !G2);}
#define S2 {assert(!G1 && G2);}
#define S3 {assert(G1 && G2);}

#define T12 T1b
#define T13 T1a
#define T21 T2b
#define T23 T2a
#define T31 T2c
#define T32 T1c
#define T33a T1d
#define T33b T2d

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();

	if (G1 && !G2) {
		
		while (1) {
			S1;
			if (flip()) {
				T12; S2;
				T21; S1;
			}
			else {
				if (flip()) {
					T12; S2;
					T23; S3;
				}
				else {
					T13; S3;
				}
				while (flip()) {
					if (flip()) {
						T33a; S3;
					}
					else if (flip()) {
						T33b; S3;
					}
					else {
						T32; S2;
						T23; S3;
					}
				}
				if (flip()) {
					T32; S2;
					T21; S1;
				}
				else {
					T31; S1;
				}
			}
		}
		
	}
	else if (!G1 && G2) {

		while (1) {
			S2;
			if (flip()) {
				T21; S1;
				T12; S2;
			}
			else {
				if (flip()) {
					T21; S1;
					T13; S3;
				}
				else {
					T23; S3;
				}
				while (flip()) {
					if (flip()) {
						T33a; S3;
					}
					else if (flip()) {
						T33b; S3;
					}
					else {
						T31; S1;
						T13; S3;
					}
				}
				if (flip()) {
					T31; S1;
					T12; S2;
				}
				else {
					T32; S2;
				}
			}
			
		}
		
	}
	else if (G1 && G2) {

		while (1) {
			S3;
			if (flip()) {
				T33a; S3;
			}
			else if (flip()) {
				T33b; S3;
			}
			else if (flip()) {
				T31; S1;
				T13; S3;
			}
			else {
				if (flip()) {
					T31; S1;
					T12; S2;
				}
				else {
					T32; S2;
				}
				while (flip()) {
					T21; S1;
					T12; S2;
				}
				if (flip()) {
					T21; S1;
					T13; S3;
				}
				else {
					T23; S3;
				}
			}
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

