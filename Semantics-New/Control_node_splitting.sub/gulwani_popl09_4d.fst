
model gulwani_popl09_4d {

	var n, m, x, y, s, c1, c2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n >= 0 && m >= 0 && x = 0 && c1 = 0 && c2 = 0 && y >= 0 && s = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x < n;
		action := s' = 2, y' = 0, x' = x + 1, c1' = c1 + 1, c2' = 0;
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
		action := s' = 1;
	};

}

strategy s {

	Region init := {state = k1};

}

