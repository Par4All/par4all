#if (defined(__STDC__) || defined(__cplusplus)) 

extern void Smith(Matrix *A, Matrix **U, Matrix **V, Matrix **Product);
extern void Hermite(Matrix *A, Matrix **H, Matrix **U);

#else 

extern void Smith(/*Matrix *A, Matrix **U, Matrix **V, Matrix **Product */);
extern void Hermite(/*Matrix *A, Matrix **H, Matrix **U */);

#endif
