
model gulwani_popl09_4b {

	var x0, y0, x, y, n, m, s, c1, c2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x0 >= 0 && y0 >= 0 && x = x0 && y = y0 && n >= 0 && m >= 0 && n >= x0 && m >= y0 && s = 1 && c1 = 0 && c2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x < n;
		action := s' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 2 && y < m;
		action := y' = y + 1, c2' = c2 + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2;
		action := s' = 1, x' = x + 1, c1' = c1 + 1, c2' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

