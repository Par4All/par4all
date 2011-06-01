
void error(void) {
	exit(1);
}

void whileif(void) {
	int x, y;
	x = y = 0;
	while (x <= 58) {
		if (y <= 8) {
			x++; y++;
			if (6 * y > x + 50) error();
		}
		else {
			x++;
			if (6 * y > x + 50) error();
		}
	}
}


void whilewhile(void) {
	int x, y;
	x = y = 0;
	while (x <= 58) {
		while (x <= 58 && y <= 8) {
			x++; y++;
			if (6 * y > x + 50) error();
		}
		while (x <= 58 && y > 8) {
			x++;
			if (6 * y > x + 50) error();
		}
	}
}

int main(void) {
	whileif();
	whilewhile();
	return 0;
}
