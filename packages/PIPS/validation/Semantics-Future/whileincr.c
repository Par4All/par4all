// This system is a reencoding of the program:
// > int i, n, a;
// > assert(i <= n);
// > while (i <= n) {
// >   i += a;
// > }
// The goal is to show there is an infinite loop if a <= 0.

// $Id$

#define USE_ASSERT 1

// usefull stuff

#include <stdlib.h>
#include <stdio.h>

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

#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c;} else freeze;}

#define G (i <= n)
#define Ga (G && i + a <= n)
#define Gb (G && i + a > n)
#define C (i += a)
#define T {trans(G, C);}
#define Ta {trans(Ga, C);}
#define Tb {trans(Gb, C);}

#define S1 {assert(i <= n);}
#define S2 {assert(i > n);}

#define SYSTEM {\
		S1;\
		while (flip()) {\
			Ta; S1;\
		}\
		Tb; S2;\
	}
	

void run(void) {
	int i, n, a;

	i = rand();
	n = rand();
	a = rand();

	if (i <= n) {
		if (a <= 0) {
			SYSTEM;
			printf("not reachable\n");
		}
		else {
			SYSTEM;
			printf("reachable\n");
		}
	}

}

int main(void) {
	run();
	return 0;
}

