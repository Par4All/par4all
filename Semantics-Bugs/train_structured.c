// TRAIN CONTROLER SYSTEM
// N. Halbwachs. Delay analysis in synchronous programs. In Proceedings of
// conference on Computer-Aided Verification, 1993, pages 333-346, 1993.

// $Id$

#define GOOD (b - s >= -10 && b - s <= 19)

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

#define check(e) {if (!(e)) checking_error();}

#define freeze {while(1);}

#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}

// transition system

#define T 1
#define L 2
#define B 3
#define S 4

#define G2 (t == T && b > s - 9)
#define U2 {s++;}
#define T2 {trans(G2, U2);}

#define G3 (t == T && b == s + 9)
#define U3 {t = B; b++; d = 0;}
#define T3 {trans(G3, U3);}

#define G4 (t == B && b == s + 1)
#define U4 {t = T; s++; d = 0;}
#define T4 {trans(G4, U4);}

#define G5 (t == L && b == s - 1)
#define U5 {t = T; b++;}
#define T5 {trans(G5, U5);}

#define G6 (t == T && b == s - 9)
#define U6 {t = L; s++;}
#define T6 {trans(G6, U6);}

#define G7 (t == L && b < s - 1)
#define U7 {b++;}
#define T7 {trans(G7, U7);}

#define G8 (t == B && d == 9)
#define U8 {t = S; b++;}
#define T8 {trans(G8, U8);}

#define G9 (t == B && b > s + 1)
#define U9 {s++;}
#define T9 {trans(G9, U9);}

#define G10 (t == B && d < 9)
#define U10 {d++; b++;}
#define T10 {trans(G10, U10);}

#define G11 (t == S && b > s + 1)
#define U11 {s++;}
#define T11 {trans(G11, U11);}

#define G12 (t == S && b == s + 1)
#define U12 {t = T; s++; d = 0;}
#define T12 {trans(G12, U12);}

#define G13 (t == T && b < s + 9)
#define U13 {b++;}
#define T13 {trans(G13, U13);}

void run(void) {
	
	int b, s, d, t;
	b = s = d = 0;
	t = T;
	
	while (1) {

		if (flip()) {
			T2;
		}
		else if (flip()) {
			T13;
		}
		else if (flip()) {
			T6;
			while (flip()) {
				T7;
			}
			T5;
		}
		else if (flip()) {
			T3;
			"reachable! T13 T13 ... T13 T3";
			while (flip()) {
				if (flip()) {
					T9;
				}
				else {
					T10;
				}
			}
			T4;
		}
		else {
			T3;
			// useful for train, unnecessary to produce the bug
			/*while (flip()) {*/
				/*if (flip()) {*/
					/*T9;*/
				/*}*/
				/*else {*/
					/*T10;*/
				/*}*/
			/*}*/
			T8;
			/*while (flip()) {*/
				/*T11;*/
			/*}*/
			T12;
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}


