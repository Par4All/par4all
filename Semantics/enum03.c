
typedef enum a {
    HS_PARSE_ONLY,
} ;

int vhs_set_r(enum a o) {
	return 1;
}

/* replacing enum a by an int works fine */
void hs_set_r() {
    enum a o;
	int res = vhs_set_r(o);
}

