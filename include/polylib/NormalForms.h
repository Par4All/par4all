#if __STDC__ 

extern void Smith(Matrix *A, Matrix **U, Matrix **V, Matrix **Product);
extern void Hermite(Matrix *A, Matrix **H, Matrix **U);

#else 

extern void Smith(/*Matrix *A, Matrix **U, Matrix **V, Matrix **Product */);
extern void Hermite(/*Matrix *A, Matrix **H, Matrix **U */);

#endif
