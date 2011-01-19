// Example of Linear Relation Analysis given by Nicolas Halbwachs in his great
// tutorial at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/programme.html

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (ok == 1)

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

#define G1 (b == 0)
#define U1 {b = 1; x++; if (ok ==  1 && x >= y) {ok = 1;} else {ok = 0;}}
#define T1 {trans(G1, U1);}

#define G2 (b == 1)
#define U2 {b = 0; y++; if (ok == 1 && x >= y) {ok = 1;} else {ok = 0;}}
#define T2 {trans(G2, U2);}

#define S1 {assert(b == 0);}
#define S2 {assert(b == 1);}

void run(void) {
	
	int b, ok, x, y;
	b = x = y = 0;
	ok = 1;
	
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
		T1;
		S2;
		T2;
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

