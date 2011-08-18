
model gulwani_cav09_3 {

	var x, y, n, i1, i2, s;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && i1 = 0 && i2 = 0 && n >= 0 && s = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x < n;
		action := s' = 2, i1' = i1 + 1, y' = x;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 2 && y < n;
		action := i2' = i2 + 1, y' = y + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2;
		action := s' = 1, x' = y + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

