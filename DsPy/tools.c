#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <complex.h>

// These macros must not be used directy !
#ifdef __PYPS_SAC_VALIDATE
#define _print_array(name, ptr, n, format, stype) \
	fwrite(ptr, n, stype, stdout); \
	fflush(stdout);
#elif defined(__PYPS_SAC_BENCHMARK)
#define _print_array(name, ptr, n, format, stype)
#else
#define _print_array(name, ptr, n, format, stype) \
	int i; \
	char formatnl[10]; \
	printf("%s :\n", name); \
	printf("----\n"); \
	formatnl[7] = 0; \
	strncpy(formatnl, format, 7); \
	strncat(formatnl, "\n", 2); \
	for (i = 0; i < n; i++) \
		printf(formatnl, *(ptr+i)); \
	printf("----\n");
#endif

static FILE* _f_data_file = 0;
int _init_data(void* ptr, const ssize_t n);


void init_data_file(const char* data_file)
{
	if (_f_data_file != 0)
		return;
	_f_data_file = fopen(data_file, "r");
	if (_f_data_file == 0)
	{
		perror("open data file");
		exit(errno);
	}
}

void close_data_file()
{
	if (_f_data_file != 0)
		fclose(_f_data_file);
}

void print_array_float(const char* name, const float* arr, const unsigned int n)
{
	_print_array(name, arr, n, "%f", sizeof(float));
}

void print_array_int(const char* name, const int* arr, const unsigned int n)
{
	_print_array(name, arr, n, "%d", sizeof(int));
}

void print_array_double(const char* name, const float* arr, const unsigned int n)
{
	_print_array(name, arr, n, "%a", sizeof(double));
}

void print_array_long(const char* name, const long* arr, const unsigned int n)
{
	_print_array(name, arr, n, "%a", sizeof(long));
}

void print_array_cplx(const char* name, const float complex* arr, const unsigned int n)
{
	int i;
	for (i=0; i<n;i++)
	{
		printf("%f %f\n",crealf(arr[i]),cimagf(arr[i]));
	}
}

int _init_data(void* ptr, const ssize_t n)
{
	ssize_t nr;
	ssize_t ntoread;

	ntoread = n;
	if (_f_data_file == 0)
	{
		fprintf(stderr, "Data file must be initialized !\n");
		exit(1);
	}
	while (ntoread > 0)
	{
		nr = fread(ptr, 1, ntoread, _f_data_file);
		if (nr == 0 && ferror(_f_data_file))
		{
			perror("read data file");
			clearerr(_f_data_file);
			return errno;
		}
		if (nr < ntoread)
		{
	//		fprintf(stderr, "%d bytes remaining...\n", ntoread-nr);
			fseek(_f_data_file, 0L, SEEK_SET);
			fflush(_f_data_file);
		}
		ntoread -= nr;
		ptr += nr;
	}

	// Old implementation... :
	//fprintf(stderr, "Warning: missing %d bytes in data file ! Filling with zeros...\n", n-nr);
	// This makes pips crashes... !!
	//memset(ptr + nr, 0, n-nr);
	return nr;
}

int init_data_gen(void* ptr, const unsigned int n, const ssize_t stype)
{
	return _init_data(ptr, (ssize_t)(n)*stype);
}

int init_data_float(float* ptr, const unsigned int n)
{
	int r = init_data_gen(ptr, n, sizeof(float));
	return r;
}

int init_data_double(double* ptr, const unsigned int n)
{
	return init_data_gen(ptr, n, sizeof(double));
}

int init_data_long(long* ptr, const unsigned int n)
{
	return init_data_gen(ptr, n, sizeof(long));
}

int init_data_int(int* ptr, const unsigned int n)
{
	return init_data_gen(ptr, n, sizeof(int));
}

int init_data_cplx(float complex* ptr, const unsigned int n)
{
	return 0;
}

void init_args(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s kernel_size data_file\n", argv[0]);
		exit(1);
	}
	init_data_file(argv[2]);
}
