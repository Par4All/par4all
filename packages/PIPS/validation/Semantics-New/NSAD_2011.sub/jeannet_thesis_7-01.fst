
model jeannet_thesis_7_01 {

	var b, c, x0, y0, x, y, z;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x0 >= 0 && y0 >= 0 && z >= 0 && x = x0 && y = y0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x + y <= -1 && z >= 0;
		action := b' = 1, c' = 1, x' = x + 1, y' = y + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x + y <= -1 && z < 0;
		action := b' = 1, c' = 0, x' = x + 1, y' = y + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := x + y > -1;
		action := b' = 0, c' = 0, x' = x - 1, y' = y - 1;
	};

}

strategy s {

	Region init := {state = k1};

}

