// Parametric version of maisonneuve17-3
// s1: size of first square
// s2: size of second square
// o: size of overlappping square

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD (x == 0 && y == s1 + s2 - o)

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

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

// transition system

#define G1 (x >= 0 && x <= s1 && y >= 0 && y <= s1)
#define U1 {x++; y++;}
#define T1 {trans(G1, U1);}

#define G2 (x >= s1 - o && x <= s1 + s2 - o && y >= s1 - o && y <= s1 + s2 - o)
#define U2 {x--; y--;}
#define T2 {trans(G2, U2);}

#define T1a_LO {trans(G1 && x <= s1 - o - 2 && y == s1, U1);}
#define T1b_LO {trans(G1 && x == s1 && y <= s1 - o - 2, U1);}
#define T1_LL {trans(G1 && x + 1 >= 0 && x + 1 <= s1 && y + 1 >= 0 && y + 1 <= s1 && y + 1 < s1 + s2 - o - x - 1, U1);}
#define T1_LB {trans(G1 && x + 1 >= s1 - o && x + 1 <= s1 + s2 - o && y + 1 >= s1 - o && y <= s1 + s2 - o && y + 1 >= s1 + s2 - o - x - 1, U1);}
#define T1_BB T1_LB

#define T2a_BO {trans(G2 && x == s1 - o && y >= s1 - o + 2, U2);}
#define T2b_BO {trans(G2 && x >= s1 - o + 2 && y == s1 - o, U2);}
#define T2_LL T2_BL
#define T2_BL {trans(G2 && x - 1 >= 0 && x - 1 <= s1 && y - 1 >= 0 && y - 1 <= s1 && y - 1 < s1 + s2 - o - x + 1, U2);}
#define T2_BB {trans(G2 && x - 1 >= s1 - o && x - 1 <= s1 + s2 - o && y - 1 >= s1 - o && y - 1 <= s1 + s2 - o && y - 1 >= s1 + s2 - o - x + 1, U2);}

#define S_L {assert(G1 && y < s1 + s2 - o - x);}
#define S_B {assert(G2 && y >= s1 + s2 - o - x);}

void run(void) {
	
	int s1, s2, o, x, y;
	s1 = rand();
	s2 = rand();
	o = rand();
	x = rand();
	y = rand();
	if (o >= 0 && s1 >= o + 3 && s2 >= o + 3 && G1 && y <= 5 - x) {
		
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

