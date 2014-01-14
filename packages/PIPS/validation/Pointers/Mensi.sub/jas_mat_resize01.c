/******************************************************************************\
* Types.
\******************************************************************************/
typedef int int_fast32_t;
/* An element in a sequence. */
typedef int_fast32_t jas_seqent_t;

/* An element in a matrix. */
typedef int_fast32_t jas_matent_t;

/* Matrix. */

typedef struct {

	/* Additional state information. */
	int flags_;

	/* The starting horizontal index. */
	int_fast32_t xstart_;

	/* The starting vertical index. */
	int_fast32_t ystart_;

	/* The ending horizontal index. */
	int_fast32_t xend_;

	/* The ending vertical index. */
	int_fast32_t yend_;

	/* The number of rows in the matrix. */
	int_fast32_t numrows_;

	/* The number of columns in the matrix. */
	int_fast32_t numcols_;

	/* Pointers to the start of each row. */
	jas_seqent_t **rows_;

	/* The allocated size of the rows array. */
	int_fast32_t maxrows_;

	/* The matrix data buffer. */
	jas_seqent_t *data_;

	/* The allocated size of the data array. */
	int_fast32_t datasize_;

} jas_matrix_t;



int jas_matrix_resize(jas_matrix_t *matrix, int numrows, int numcols)
{
	int size;
	int i;

	size = numrows * numcols;
	if (size > matrix->datasize_ || numrows > matrix->maxrows_) {
		return -1;
	}

	matrix->numrows_ = numrows;
	matrix->numcols_ = numcols;

	//for (i = 0; i < numrows; ++i) {
		matrix->rows_[i] = &matrix->data_[numcols * i];
		//}

	return 0;
}
