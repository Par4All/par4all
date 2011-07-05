
model jeannet_thesis_8_05 {

	var s, x;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && x = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x <= 9;
		action := x' = x + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x >= 10;
		action := s' = 2, x' = x + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 2 && x <= 5;
		action := s' = 1, x' = x - 1;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := s = 2 && x >= 6;
		action := x' = x - 1;
	};

}

strategy s {

	Region init := {state = k1};

}

