#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s nb_floats\n", argv[0]);
		return EXIT_FAILURE;
	}

	int i;
	int n = atoi(argv[1]);
	unsigned int seed;
	int frand = open("/dev/urandom", O_RDONLY);
	read(frand, &seed, sizeof(unsigned int));
	close(frand);

	srand(seed);
	for (i=0; i < n; i++) {
		float f = (float) rand();
		fwrite(&f, sizeof(float), 1, stdout);
	}
	fflush(stdout);

	return EXIT_SUCCESS;
}
