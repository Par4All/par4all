
model gonnord_thesis_4_09 {

	var s, i, j;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && i = 0 && j = -100;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && i <= 100;
		action := s' = 2, i' = i + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 2 && j <= 19;
		action := j' = j + i;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2;
		action := s' = 1;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := s = 1 && i > 100;
		action := s' = 3;
	};

}

strategy s {

	Region init := {state = k1};

}

