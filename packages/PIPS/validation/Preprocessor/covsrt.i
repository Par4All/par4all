void covsrt(covar,ma,lista,mfit)
float **covar;
int ma,lista[],mfit;
{
	int i,j;
	float swap;

	for (j=1;j<ma;j++)
		for (i=j+1;i<=ma;i++) covar[i][j]=0.0;
	for (i=1;i<mfit;i++)
		for (j=i+1;j<=mfit;j++) {
			if (lista[j] > lista[i])
				covar[lista[j]][lista[i]]=covar[i][j];
			else
				covar[lista[i]][lista[j]]=covar[i][j];
		}
	swap=covar[1][1];
	for (j=1;j<=ma;j++) {
		covar[1][j]=covar[j][j];
		covar[j][j]=0.0;
	}
	covar[lista[1]][lista[1]]=swap;
	for (j=2;j<=mfit;j++) covar[lista[j]][lista[j]]=covar[1][j];
	for (j=2;j<=ma;j++)
		for (i=1;i<=j-1;i++) covar[i][j]=covar[j][i];
}
