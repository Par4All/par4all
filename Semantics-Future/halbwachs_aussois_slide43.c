// Example of Linear Relation Analysis given by Nicolas Halbwachs in his great
// tutorial at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/programme.html

// $Id$

#define USE_ASSERT 0
#define GOOD (t >= 0 && s >= 0 && s <= 4 && d >= 0 && d <= 4 * t + s)

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

#define G1 (s <= 3)
#define G1a (s <= 2)
#define G1b (s == 3)
#define C1 {s++; d++;}
#define T1 {trans(G1, C1);}
#define T1a {trans(G1a, C1);}
#define T1b {trans(G1b, C1);}

#define G2 (1)
#define C2 {t++; s = 0;}
#define T2 {trans(G2, C2);}

#define S1 {assert(s <= 3);}
#define S2 {assert(s >= 4);}

void run(void) {
	int t, d, s;
	t = d = s = 0;
	
	while (1) {
		
		S1;
		if (flip()) {
			T1a; S1;
		}
		else if (flip()) {
			T2; S1;
		}
		else {
			T1b; S2;
			T2; S1;
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

