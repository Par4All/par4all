
/*#define USE_ASSERT*/

#ifdef USE_ASSERT
#include <assert.h>
#else
#define assert(c) {}
#endif

#define check(c) if (!(c)) error()
#define check_not(c) if (c) error()

/*#define VERBOSE*/

#ifdef VERBOSE
#include <stdio.h>
#define display(c) {printf(c); printf("\n"); fflush(stdout);}
#else
#define display(c)
#endif

#include <stdlib.h>

void error() {
	exit(1);
}

int alea() {
	return rand() % 2;
}

#define CP_VALS {e_ = e; ne_ = ne; uo_ = uo; i_ = i;}

#define G_RM (i >= 1)
#define U_RM {display("RM"); check(G_RM); CP_VALS; e = 0; ne = e_ + ne_; uo = uo_ + 1; i = i_ - 1;}
#define G_WH (ne + uo >= 1)
#define U_WH {display("WH"); check(G_WH); CP_VALS; e = e_ + 1; ne = 0; uo = 0; i = ne_ + uo_ + i_ - 1;}
#define G_WM (i >= 1)
#define U_WM {display("WM"); check(G_WM); CP_VALS; e = 1; ne = 0; uo = 0; i = e_ + ne_ + uo_ + i_ - 1;}

/*
void run_() {
	
	int i0 = 2;
	
	int e = 0, ne = 0, uo = 0, i = i0;
	int e_, ne_, uo_, i_;
	
	assert(e == 0 && ne == 0 && uo == 0 && i >= 1);
	if (alea()) {
		U_RM;
	}
	else {
		U_WM;
		assert(e == 1 && ne == 0 && uo == 0 && i >= 0);
		while (alea() && G_WM) {
			U_WM;
			assert(e == 1 && ne == 0 && uo == 0 && i >= 0);		
		}
		if (G_RM) {
			U_RM;
		}
		else {
			return;
		}
	}
	
	while (1) {
		assert(e == 0 && ne <= 1 && uo >= 1 && i >= 0);
		while (alea() && G_RM) {
			U_RM;
			assert(e == 0 && ne <= 1 && uo >= 1 && i >= 0);
		}
		while (alea()) {
			if (alea() && G_WM) {
				U_WM;
			}
			else {
				U_WH;
			}
			assert(e == 1 && ne == 0 && uo == 0 && i >= 0);
			while (alea() && G_WM) {
				U_WM;
				assert(e == 1 && ne == 0 && uo == 0 && i >= 0);
			}
			if (G_RM) {
				U_RM;
			}
			else {
				return;
			}
		}
	}
	
}
*/

void run() {
	
	//int i0 = 1 + rand();
	int i0 = 5;
	
	int e = 0, ne = 0, uo = 0, i = i0;
	int e_, ne_, uo_, i_;

	assert(e == 0 && ne == 0 && uo == 0 && i == i0);

	if (i0 == 1) {
		check_not(G_WH);
		if (alea()) {
			U_RM;
			assert(e == 0 && ne == 0 && uo == 1 && i == 0);
			check_not(G_RM);
			check_not(G_WM);
			U_WH;
		}
		else {
			U_WM;
		}
		assert(e == 1 && ne == 0 && uo == 0 && i == 0);
	}
	
	else {
	
		check_not(G_WH);
		if (alea()) {
			U_RM;
			assert(e == 0 && ne == 0 && uo != 0 && i != 0 && uo + i == i0);
			while (alea() && G_RM) {
				U_RM;
				assert(e == 0 && ne == 0 && uo + i == i0);
			}
			if (i != 0) {
				if (alea()) {
					U_WH;
				}
				else {
					U_WM;
				}
			}
			else {
				check_not(G_RM);
				check_not(G_WM);
				U_WH;
			}
		}
		else {
			U_WM;
		}
		
		while (1) {
			assert(e == 1 && ne == 0 && uo == 0 && i == i0 - 1);
			check_not(G_WH);
			while (alea()) {
				U_WM;
			}
			while (alea()) {
				U_RM;
				assert(e == 0 && ne == 1 && uo + i == i0 - 1);
				while (alea() && G_RM) {
					U_RM;
					assert(e == 0 && ne == 1 && uo + i == i0 - 1);
				}
				if (i != 0) {
					if (alea()) {
						U_WH;
					}
					else {
						U_WM;
					}
				}
				else {
					check_not(G_RM);
					check_not(G_WM);
					U_WH;
				}
			}
		}
	
	}
	
}

int main(void) {
	run();
	return 0;
}

