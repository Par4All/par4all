
model halbwachs_thesis_3_01 {

	var i, j;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := i = 0 && j = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i <= 100;
		action := i' = i + 4;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := i <= 100;
		action := i' = i + 2, j' = j + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

