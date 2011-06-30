
model jeannet_thesis_8_04 {

	var s, x, y, e, f;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && x = 0 && y = 0 && e >= 0 && e <= 1 && f >= 0 && f <= 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && e = 1;
		action := s' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && e = 0;
		action := s' = 3;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2 && e = 1;
		action := x' = x + 1, y' = y + f;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := s = 2 && e = 0;
		action := s' = 3;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := s = 3;
		action := s' = 2, x' = x + 1, y' = y + 2;
	};

}

strategy s {

	Region init := {state = k1};

}

