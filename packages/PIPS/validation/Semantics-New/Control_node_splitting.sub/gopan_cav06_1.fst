
model gopan_cav06_1 {

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
		guard := s = 1 && x <= 50;
		action := s' = 2, y' = y + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x > 50;
		action := s' = 2, y' = y - 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2 && y >= 0;
		action := s' = 1, x' = x + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

