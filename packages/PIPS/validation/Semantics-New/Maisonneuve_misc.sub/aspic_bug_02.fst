// Fails with Aspic v3.1 (compiled on Monday 15 March 2010)
// during acceleration:
// Fatal error: exception Invalid_argument("Vector.set: index")

model aspic_bug_02 {

	var x, y, z;
	states k;

	transition t1 := {
		from := k;
		to := k;
		guard := x >= 2 * z;
		action := x' = x + 1, y ' = y + 1, z' = z + 1;
	};

	transition t2 := {
		from := k;
		to := k;
		guard := true;
		action := z' = 0;
	};

}

strategy s {

	Region init := {state = k && x = 0 && y = 0 && z = 0};

}

