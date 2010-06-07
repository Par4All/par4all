#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "newgen_types.h"
#include "newgen_hash.h"

#define GAP_BETWEEN_WORDS (0)

int main(argc, argv)
int argc;
char *argv[];
{
    hash_table hwords;

    char word[32], **twords;
    FILE *fd;
    int nwords, iwords;

    if (argc != 2)  {
	fprintf(stderr, "Usage: %s number-of-words\n", argv[0]);
	exit(1);
    }

    nwords = atoi(argv[1]);
    twords = (char **) malloc(nwords*sizeof(char *));

    if ((fd = fopen("/usr/dict/words", "r")) == NULL) {
	fprintf(stderr, "Cannot open /usr/dict/words\n");
	exit(1);
    }

    iwords = 0;
    while (fscanf(fd, "%s", word) != EOF) {
	int word_gap;

	twords[iwords++] = strdup(word);
	if (iwords == nwords)
	    break;
	/* skip some words */
	for (word_gap = 0; word_gap < GAP_BETWEEN_WORDS; word_gap++)
	    if (fscanf(fd, "%s", word) == EOF)
		break;
    }

    fclose(fd);

    if (iwords != nwords) {
	fprintf(stderr, "Not that many words in /usr/dict/words\n");
	exit(1);
    }

    hwords = hash_table_make(hash_string, 0);

    fprintf(stderr, "step 01\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 02\n");
    for (iwords = 2; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 03\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 1\n");
	    exit(1);
	}
    }

    fprintf(stderr, "step 04\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 05\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 06\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 2\n");
	    exit(1);
	}
    }

    fprintf(stderr, "step 07\n");
    for (iwords = 3; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 08\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 09\n");
    for (iwords = 0; iwords < nwords; iwords += 1) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 3\n");
	    exit(1);
	}
    }

    hash_table_print(hwords);

    hash_table_clear(hwords);

    fprintf(stderr, "step 11\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 12\n");
    for (iwords = 2; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 13\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 1\n");
	    exit(1);
	}
    }

    fprintf(stderr, "step 14\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 15\n");
    for (iwords = 0; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 16\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 2\n");
	    exit(1);
	}
    }

    fprintf(stderr, "step 17\n");
    for (iwords = 3; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 18\n");
    for (iwords = 1; iwords < nwords; iwords += 4) {
	hash_put(hwords, twords[iwords], (char *) iwords);
    }

    fprintf(stderr, "step 19\n");
    for (iwords = 0; iwords < nwords; iwords += 1) {
	if ((int) hash_del(hwords, twords[iwords]) != iwords) {
	    fprintf(stderr, "error 3\n");
	    exit(1);
	}
    }

    hash_table_print(hwords);

    hash_table_clear(hwords);

    return 0;
}
