// LEAKING GAZ BURNER
// Zhou Chaochen, C. A. R. Hoare, Anders P. Ravn: A Calculus of Durations. Inf.
// Process. Lett. 40(5): 269-276 (1991)

// $Id$

#define USE_ASSERT 0
#define BAD (6 * l > t + 50)

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

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

enum {
	L = 0,
	N = 1
};

#define G1 (s == L && x <= 9)
#define G2 (s == L)
#define G3 (s == N)
#define G4 (s == N && x >= 50)

#define C1 {t++; l++; x++;}
#define C2 {x = 0; s = N;}
#define C3 {t++; x++;}
#define C4 {x = 0; s = L;}

void run(void) {
	int s;
	int l, t, x;
	s = 0;
	l = t = x = 0;
	
	check_not(BAD);
	while (1) {
		assert(s == L && x <= 9);
		while (flip()) {
			trans(x <= 8 && G1, C1);
			assert(s == L && x <= 9);
		}
		if (flip()) {
			trans(x == 9 && G1, C1);
			assert(s == L && 10 <= x && x <= 49);
		}
		trans(G2, C2);
		assert(s == N && x <= 9);
		while (flip()) {
			trans(x <= 8 && G3, C3);
			assert(s == N && x <= 9);
		}
		trans(x == 9 && G3, C3);
		assert(s == N && 10 <= x && x <= 49);
		while (flip()) {
			trans(x <= 48 && G3, C3);
			assert(s == N && 10 <= x && x <= 49);
		}
		trans(x == 49 && G3, C3);
		assert(s == N && x >= 50);
		while (flip()) {
			trans(G3, C3);
			assert(s == N && x >= 50);
		}
		trans(G4, C4);
	}
		
}

int main(void) {
	run();
	return 0;
}

