
model henzinger_popl02_1 {

	var s, lock, x, y;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := lock = 0 && x >= 0 && y >= 0 && x != y && s = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x != y;
		action := lock' = 1, x' = y;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x != y;
		action := lock' = 0, x' = y, y' = y + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 1 && x = y;
		action := s' = 2;
	};

}

strategy s {

	Region init := {state = k1};

}

