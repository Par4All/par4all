
model jeannet_thesis_7_02 {

	var s, b, x;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && b >= 0 && b <= 1 && x >= 0 && x <= 5;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && x <= 2 && b = 1;
		action := b' = 1, x' = x + 10, s' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x <= 2 && b = 0;
		action := x' = x - 1, s' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && x >= 3;
		action := b' = 0, x' = x - 1, s' = 2;
	};

}

strategy s {

	Region init := {state = k1};

}

