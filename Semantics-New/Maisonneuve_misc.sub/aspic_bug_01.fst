// Fails with Aspic v3.1 (compiled on Monday 15 March 2010).
// if acceleration is enabled (default case):
// says bad location is unreachable, while it is clearly reachable.

model aspic_bug_01 {

	var s;
	states k, l;

	transition t := {
		from := k; // works if we go from k to l
		to := k;   // instead of looping on k
		guard := s = 1; // works if guard is true instead of s=1
		action := s' = 2;
	};

}

strategy s {

	Region init := {state = k && s = 1};
	Region bad := {s = 2};

}

