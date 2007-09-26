#if defined(__cplusplus)
extern "C" {
#endif

extern void Smith(Matrix *A, Matrix **U, Matrix **V, Matrix **Product);
extern void Hermite(Matrix *A, Matrix **H, Matrix **U);

#if defined(__cplusplus)
}
#endif
