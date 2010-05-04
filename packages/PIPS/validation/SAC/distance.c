double distance(double X[2], double Y[2])
{
    double tmp;
    tmp = cos(X[0])*cos( Y[0] ) * cos(X[1]-Y[0]) + sin(X[0])*sin(Y[0]);
    return 6368.* acos(tmp);
}
