
model gonnord_thesis_6_08 {

	var s, x, y;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && x = 0 && y = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x <= 100;
		action := x' = x + 1, y' = y + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x <= 100;
		action := x' = x + 2;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 1;
		action := s' = 2;
	};

}

strategy s {

	Region init := {state = k1};

}

