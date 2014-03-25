Ac = [0.4990, -0.0500; 0.0100, 1.0000];
Bc = [1; 0];
Cc = [564.48, 0];
Dc = -1280;
xc = zeros(2, 1);
receive(y, 2); receive(yd, 3);
while (1)
    yc = max(min(y - yd, 1), -1);
    skip;
    u = Cc*xc + Dc*yc;
    xc = Ac*xc + Bc*yc;
    send(u, 1);
    receive(y, 2);
    receive(yd, 3);
    skip;
end
