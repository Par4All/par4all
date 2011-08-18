
model jeannet_thesis_7_05 {

	var b, x;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x < 0 && b >= 0 && b <= 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := b = 1;
		action := x' = x + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := b = 0;
		action := x' = x - 1;
	};

}

strategy s {

	Region init := {state = k1};

}

