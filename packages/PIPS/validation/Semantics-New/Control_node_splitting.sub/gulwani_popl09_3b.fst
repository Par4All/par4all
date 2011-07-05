
model gulwani_popl09_3b {

	var n, x, c, s;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n >= 0 && x = 0 && c = 0 && s = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x < n;
		action := x' = x + 1, c' = c + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1;
		action := s' = 2;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2 && x < n;
		action := x' = x + 1, c' = c + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

