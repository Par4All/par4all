$(function(){

    //
    // LEFT COLUMN
    //

    // A+ / A- buttons
    $('#aplus').click(function(ev) {
	ev.preventDefault();
	resize(1)
    });
    $('#aminus').click(function(ev) {
	ev.preventDefault();
	resize(0)
    });

    // Load classic examples button
    $('#classic-examples-dialog a').click(function(ev) {
	ev.preventDefault();
	load_example(ev.target.id);
    });

    // Upload file button
    $('#upload_input').change(function() {
	$('#upload_form').submit();
	$('#upload_target').load(after_upload);
    });

    // Run button
    $("#run-button").click(function(ev) {
	ev.preventDefault();
	activate_tab('result');
	if (performed == false)
	    performed = perform_operation(1, 'result');
    });

    // Print buttons
    $("#print-button").click(function(ev) {
        var content    = $('#resultcode'),
            printFrame = $('#iframetoprint').contentWindow;
        printFrame.document.ope();
        printFrame.document.write(content.innerHTML);
        printFrame.document.close();
        printFrame.focus();
        printFrame.print();
    });

    // Switch between basic/advanced modes
    $('#mode-buttons').click(function(ev) {
	ev.preventDefault();
	advanced = !advanced;
	switch_adv_mode();
    });

    // Initialize tooltips for advance mode form
    $('#adv-form a').tooltip();

    // Initialize button states
    switch_buttons();
    switch_adv_mode();


    //
    // MAIN COLUMN
    //

    // Source code input field
    $('#sourcecode-1').linedtextarea()
	.attr('spellcheck', false);

    // "RESULTS" tab
    $('#result_tab a').click(function(ev) {
	ev.preventDefault();
	if (performed == false)
	    performed = perform_operation(1, 'result');
    });
    // "GRAPH" tab
    $('#graph_tab a').click(function(ev) {
	ev.preventDefault();
	if (created_graph == false)
	    created_graph = create_graph(1, 'graph');
    });


});
