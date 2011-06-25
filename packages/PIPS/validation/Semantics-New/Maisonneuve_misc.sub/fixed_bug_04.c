// Poor analysis of multiplication in deprecated versions of PIPS.

// $Id$

void run(void) {
	int i, n, m;
	i = rand();
	n = rand();
	m = rand();
	
	if (m >= 0 && n >= 1 && i <= m) {
		if (i > n * m) "unreachable";
	}
	
}

int main(void) {
	run();
	return 0;
}

