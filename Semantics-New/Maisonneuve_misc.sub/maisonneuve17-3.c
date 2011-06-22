// Same as maisonneuve17-1, restructured by partitioning the union of
// transition guards into two convex sets.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD1 (x == 2 && y == 7)
#define BAD2 (x == 5 && y == 7)

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

#define trans(g, c) {if (g) {c; check_not(BAD1); check_not(BAD2);} else freeze;}

// transition system

#define G1 (x >= 0 && x <= 5 && y >= 0 && y <= 5)
#define U1 {x++; y++;}
#define T1 {trans(G1, U1);}

#define G2 (x >= 4 && x <= 9 && y >= 4 && y <= 9)
#define U2 {x--; y--;}
#define T2 {trans(G2, U2);}

#define T1a_LO {trans(G1 && x <= 2 && y == 5, U1);}
#define T1b_LO {trans(G1 && x == 5 && y <= 2, U1);}
#define T1_LL {trans(G1 && x <= 4 && y <= 4 && y <= 6 - x, U1);}
#define T1_LB {trans(G1 && x >= 3 && y >= 3 && y >= 7 - x, U1);}
#define T1_BB {trans(G1 && x <= 5 && y <= 5, U1);}

#define T2a_BO {trans(G2 && x == 5 && y >= 7, U2);}
#define T2b_BO {trans(G2 && x >= 7 && y == 5, U2);}
#define T2_LL {trans(G2 && x == 4 && y == 4, U2);}
#define T2_BL {trans(G2 && x <= 6 && y <= 6 && y <= 10 - x, U2);}
#define T2_BB {trans(G2 && x >= 5 && y >= 5 && y >= 11 - x, U2);}

#define S_L {assert(G1 && y <= 8 - x);}
#define S_B {assert(G2 && y >= 9 - x);}

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();
	if (G1 && y <= 5 - x) {
		
		while (1) {
			S_L;
			if (flip()) {
				T1a_LO; freeze;
			}
			else if (flip()) {
				T1b_LO; freeze;
			}
			else if (flip()) {
				T1_LL; S_L;
			}
			else if (flip()) {
				T2_LL; S_L;
			}
			else {
				T1_LB; S_B;
				while (flip()) {
					if (flip()) {
						T1_BB; S_B;
					}
					else {
						T2_BB; S_B;
					}
				}
				if (flip()) {
					T2a_BO;
				}
				else if (flip()) {
					T2b_BO;
				}
				T2_BL; S_L;
			}
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

