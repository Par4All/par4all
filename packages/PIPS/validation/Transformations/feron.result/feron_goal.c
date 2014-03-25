int main(int argc, char* argv[]) {
    double xc[2], xb[2], y, yd, yc, u;
    xc[0] = 0;
    xc[1] = 0;
    receive(y, 2);
    receive(yd, 3);
    yc = y - yd;
    while (1) {
        if (yc > 1) {
            yc = 1;
        }
        if (yc < 1) {
            yc = -1;
        }
        skip;
        u = 564.48*xc[0] - 1280*yc;
        xb[0] = xc[0];
        xb[1] = xc[1];
        xc[0] = 0.4990*xb[0] - 0.0500*xb[1] + yc;
        xc[1] = 0.01*xb[0] + xb[1];
        send(u, 1);
        receive(y, 2);
        receive(yd, 3);
        yc = y - yd;
        skip;
    }
}
