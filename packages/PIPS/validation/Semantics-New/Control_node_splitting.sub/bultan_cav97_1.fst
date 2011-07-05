
model bultan_cav97_1 {

	var s1, s2, a, b;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s1 = 1 && s2 = 1 && a = 0 && b = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s1 = 1;
		action := a' = b + 1, s1' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s1 = 2 && a < b;
		action := s1' = 3;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s1 = 2 && b = 0;
		action := s1' = 3;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := s1 = 3;
		action := a' = 0, s1' = 1;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := s2 = 1;
		action := b' = a + 1, s2' = 2;
	};

	transition t6 := {
		from := k2;
		to := k2;
		guard := s2 = 2 && b < a;
		action := s2' = 3;
	};

	transition t7 := {
		from := k2;
		to := k2;
		guard := s2 = 2 && a = 0;
		action := s2' = 3;
	};

	transition t8 := {
		from := k2;
		to := k2;
		guard := s2 = 3;
		action := b' = 0, s2' = 1;
	};

}

strategy s {

	Region init := {state = k1};

}

