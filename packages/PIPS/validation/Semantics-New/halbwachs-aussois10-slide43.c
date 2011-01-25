// Example of Linear Relation Analysis given by Nicolas Halbwachs in his great
// tutorial at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/programme.html

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
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

#define G1 (s <= 3)
#define G1a (s <= 2)
#define G1b (s == 3)
#define U1 {s++; d++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (1)
#define U2 {t++; s = 0;}
#define T2 {trans(G2, U2);}

#define S1 {assert(s <= 3);}
#define S2 {assert(s >= 4);}

void run(void) {
	
	int t, d, s;
	t = d = s = 0;
	
	while (1) {

#if RESTRUCTURE == 0
		
		if (flip()) {
			T1;
		}
		else {
			T2;
		}
		
#else
		
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
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

