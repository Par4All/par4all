
model gulwani_popl09_3d {

	var n, m, x, y, c;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n >= 0 && m >= 0 && n >= m && x = 0 && y = 0 && c = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x < n;
		action := x' = x + 1, y' = y  + 1, c' = c + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := y < m;
		action := x' = x + 1, y' = y + 1, c' = c + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

