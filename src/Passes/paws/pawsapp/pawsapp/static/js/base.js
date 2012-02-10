// Global vars
var directory,
    advanced = false,
    loaded = false,
    performed = false,
    created_graph = false,
    functions = '',
    nb_files = 1,
    multiple = false,
    options = {
	zoomHeight: 400,
	zoomWidth: 400
    };


/*********************************************************************
*								     *
* Helper functions for keeping consistency between input and output. *
*								     *
*********************************************************************/

function activate_tab(id) {

    $link = $('#' + id + '_tab a', '#op-tabs'); // <a> element

    // Activate selected tab
    $('li', '#op-tabs').removeClass('active');
    $link.parent().addClass('active');

    // Activate corresponding panel
    $('.tab-pane', '.tab-content').removeClass('active');
    $($link.attr('href')).addClass('active');
}

function change_tab_focus_res(result_tab_index, panel_id) {
    //$('#tabs').tabs("select", result_tab_index);
    add_wait_notification(panel_id);
}

function add_wait_notification(panel_id) {
    var tab = '#resultcode';
    if (panel_id == 'graph')
	tab = '#graph';
    $(tab).html('<div class="alert alert-info"><b>Please wait while processing...</b> It might take a long time.</div>');
}

function add_choose_function_notification() {
    $('#resultcode').html("<p><b>Choose function to display.</b><br/></p>");
}

function switch_enable(sel, value) {
    $(sel).attr('disabled', !value);
}

function switch_buttons() {
    switch_enable('#run-button', loaded && !performed);
    switch_enable('#save-button', performed);
    switch_enable('#print-button', performed);
}

function switch_adv_mode() {
    if (advanced) {
	$('#adv-form').fadeIn();
	$('#adv-button').button('toggle');
    } else {
	$('#adv-form').hide();
	$('#basic-button').button('toggle');
    }
    $('#adv-button').attr('disabled', advanced);
    $('#basic-button').attr('disabled', !advanced);
    performed = false;
    graph_created = false;
    switch_buttons();
}

function deactivate_graph_buttons() {
    //console.debug('DEACTIVATING');
}

function activate_graph_buttons() {
    //console.debug('ACTIVATING');
}

function clear_graph_tab() {
    deactivate_graph_buttons();
    add_wait_notification('graph');
}

// Delete extra source tabs and panels
function clear_multiple() {
    activate_tab('source-1');
    var tabs = $('ul#op-tabs li');
    for (var i = 0; i < tabs.length; i++)
	if (tabs[i].id.search('source-')==0 && tabs[i].id.search('source-1')==-1) {
	    $($('#' + tabs[i].id + ' a').attr('href')).remove();
	    $('#' + tabs[i].id).remove();
	}
    nb_files = 1;
    multiple = false;
    $('#multiple-functions').html('');
}


/*********************************************************************
*								     *
* Functions for loading source files.				     *
*								     *
*********************************************************************/

function load_files(files) {

    clear_multiple();

    nb_files = files.length;

    for (var i=1; i<= nb_files; i++) {

	var fname = files[i-1][0],
	    code  = files[i-1][1];

	// Create tab and panel on the fly
	if (i > 1) {
	    $("#source-" + (i-1) + "_tab").after($('#source_tab_skel').html().replace(/__skel__/g, i));
	    $('#source-' + (i-1)).after($('#source_panel_skel').html().replace(/__skel__/g, i));
	    $('#sourcecode-' + i).attr('spellcheck', false);
	    //$('#sourcecode-' + i).linedtextarea();
	}
	$("#source-" + i + "_tab a").text(fname);
	$('#sourcecode-' + i).text(code);
	language_detection(code, i);
    }

    multiple      = (nb_files > 1);
    loaded        = true;
    performed     = false;
    created_graph = false;
    switch_buttons();
}

function load_example(name) {

    var url = routes['load_example_file'].replace('{tool}', operation).replace('{name}', name);

    $('#classic-examples-dialog').modal('hide');
    $.get(url, function(data) {
	$('#pseudotextfile').val('');
	load_files([[name, data]]);
    });
}

function after_upload() {

    var result = $('#upload_target')[0].contentWindow.document.getElementsByTagName('pre')[0],
	files  = $.parseJSON($(result).text());

    $('#pseudotextfile').val($('#upload_input').val());
    load_files(files);
}


/*********************************************************************
*								     *
* Helper functions for performing operations.			     *
*								     *
*********************************************************************/

function language_detection(source, number) {
    var test;
    $.ajax({
        type: "POST",
	data: {code: source},
	cache: false,
	url: routes['detect_language'],
        async: false,
        error: function() {
            $('#language_detection').html("Language not recognised!");
            test = false;
	},
        success: function(data) {
	    var $elem = $('#lang-' + number);
	    $elem.removeClass();	    
            if (data != "none") {
		$elem.html(data);
		$elem.addClass('label label-success')
                test = true;
	    } else {
		$elem.html("not supported");
		$elem.addClass('label label-warning')
                test = false;
            }
        }
    });
    return test;
}

function compile(source, number) {
    var test;
    $.ajax({
        type: "POST",
        data: { code: source, language: $('#lang-' + number).html() },
        cache: false,
        url: routes['compile'],
        async: false,
        error: function() {
            $('#language_detection').html("Cannot be compiled!");
            test = false;
	},
        success: function(data) {
            if (data != "0") {
                $('#resultcode').html('Error in tab number ' + number + ':<br/><br/>' + data);
                test = false;
            } else {
                test = true;
            }
	}
    });
    return test;
}
	
function preprocess_input(source, result_tab_index, tab_index, panel_id) {
    if (language_detection(source, tab_index)) {
	change_tab_focus_res(result_tab_index, panel_id);
	if (compile(source, tab_index)) {
	    return true;
	}
    }
    return false;
}
	
function check_language(index) {
    var selected = $('#tabs').tabs('option', 'selected') + 1;
    var sourcee  = $('#sourcecode-' + selected).val();
    language_detection(sourcee, selected);
}

function get_directory(panel_id) {
    $.ajax({
	type: 'GET',
	cache: false,
	url: routes['get_directory'],
	async: false,
	error: function() {
	    if (panel_id == 'result')
		$('#resultcode').html('Web error, try again!');
	},
	success: function(data) {
	    directory = data;
	    if (panel_id == 'result')
		$('#save-button').click(function(ev) {
		});
	}
    });
}
 	
function get_functions() {

    var datas = {};
    datas['number'] = nb_files;
    for(var i=0; i<nb_files; i++) {
	datas['code' + i] = $('#sourcecode-' + (i+1)).text();
	datas['lang' + i] = $('#lang-' + (i+1)).html();
    }
    $.post(
	routes['get_functions'],
	datas,
	function(data) {
	    functions = data;
	    $('#choose-function-dialog .modal-body').html(data);
	    $('#choose-function-dialog input:submit').click(function() {
		$.post(
		    routes['perform_multiple'],
		    { functions: $(this).attr('value'),
		      operation: operation
		    },
		    function(data) {
                        $('#resultcode').html(data);
			switch_buttons();
		    });
		$('#dialog-choose-function').modal('hide');
	    });
	});
}
	
function check_sources(index, panel_id) {

    var sources   = new Array(nb_files),
	languages = new Array(nb_files);

    get_directory(panel_id);
    
    for (var i = 1; i <= nb_files; i++) {
	sources[i] = $('#sourcecode-' + i).text();
    }
    var u = 1;
    while ((u <= nb_files) && preprocess_input(sources[u], index, u, panel_id)) {
	u = u + 1;
    }
    if (u == nb_files + 1) {
	get_functions();
	return true;
    }
    return false;
}

/*********************************************************************
*								     *
* Functions for creating graphics.				     *
*								     *
*********************************************************************/

function create_graph(index, panel_id) {

    if (!loaded)
	return;

    clear_graph_tab();
    if (check_sources(index, panel_id)) {
	if (multiple) {
	    $.get(
		routes['dependence_graph_multi'],		
		enable_dependence_graph		   
	    );
	} else {
	    $.post(
		routes['dependence_graph'], 
		{ code: $('#sourcecode-1').text(),
                  lang: $('#lang-1').html()
                },
		enable_dependence_graph
	    );
	}
    }
    return true; //TODO
}

function enable_dependence_graph(data) {
    $('#graph').html(data);
    $('.ZOOM_IMAGE').jqzoom(options);
    activate_graph_buttons();
}


/*********************************************************************
*								     *
* Functions for handling special keys.				     *
*								     *
*********************************************************************/

function handle_keydown(item, event) {
    c = event.keyCode;
    if (c == 9) {
	catch_tab(item);
    }
    else if (c == 13 || c == 86) {
	setTimeout("check_language()", 1000);
    }
    performed = false;
    created_graph = false;
    switch_buttons();
}

function replaceSelection(input, replaceString) {
    if (input.setSelectionRange) {
	var selectionStart = input.selectionStart;
	var selectionEnd = input.selectionEnd;
	input.value = input.value.substring(0, selectionStart) + replaceString + input.value.substring(selectionEnd);
	if (selectionStart != selectionEnd) {
	    setSelectionRange(input, selectionStart, selectionStart + replaceString.length);
	} else {
	    setSelectionRange(input, selectionStart + replaceString.length, selectionStart + replaceString.length);
	}
    } else if (document.selection) {
	var range = document.selection.createRange();
	if (range.parentElement() == input) {
	    var isCollapsed = range.text = '';
	    range.text = replaceString;
	    if (!isColllapsed) {
		range.moveStart('character', -replaceString.length);
		range.select();
	    }
	}
    }
}

function setSelectionRange(input, selectionStart, selectionEnd) {
    if (input.setSelectionRange) {
	input.focus();
	input.setSelectionRange(selectionStart, selectionEnd);
    }
    else if (input.createTextRange) {
	var range = input.createTextRange();
	range.collapse(true);
	range.moveEnd('character', selectionEnd);
	range.moveStart('character', selectionStart);
	range.select();
    }
}

function catch_tab(item) {
    replaceSelection(item, String.fromCharCode(9));
    setTimeout("document.getElementById('" + item.id + "').focus();", 0);
    return false;
}

/*********************************************************************
*								     *
* Function for font resizing.					     *
*								     *
*********************************************************************/

function resize(direction) {
    changeFontSize('textarea', direction);
    changeFontSize('.highlight', direction);
    changeFontSize('.lineno', direction);
    changeLinesHeight();
}

function changeFontSize(id, direction) {
    size = parseInt($(id).css('font-size'));
    size = direction == 1 ? size + 1 : size - 1;
    if (size > 1) {
	$(id).css('font-size', size);
    }
}

function changeLinesHeight() {
    size = parseInt($('.lines').css('height'));
    size = size * 1.04;
    $('.lines').css('height', size);
}


/*********************************************************************
*								     *
* Functions for tool.js			                             *
*								     *
*********************************************************************/

function choose_function() {
    $('#multiple-functions').html(functions);
    $('#multiple-functions input:submit').click(function(ev) {
	$.ajax({
	    type: "POST",
	    data: { function: $(this).attr('value'),
		    operation: operation,
		    language: $('#lang-1').html()
		  },
	    cache: false,
	    url: routes['perform_multiple'],
	    error: function() {},
	    success: function(data) {
		$('#resultcode').html(data);
		switch_buttons();
	    }
	});
    });
    add_choose_function_notification();
}

function keyval(k, v) {
    return encodeURIComponent(k) + '=' + encodeURIComponent(v);
}

function get_form_values(sel) {
    var out = [];

    // INPUT fields
    var fields = $(sel + " input");
    for (var i=0; i < fields.length; i++) {
	var f = fields[i];
	if ((f.type == 'checkbox' && f.checked) || f.type == 'text' || f.type == 'hidden')
	    out.push(keyval(f.name, f.value));
    }
    // SELECT fields
    var fields = $(sel + " select");
    for (var i=0; i < fields.length; i++) {
	var f = fields[i];
	out.push(keyval(f.name, f.value))
    }

    return out;
}

function perform_operation(index, panel_id) {

    if(!loaded)
	return false;

    // Advanced mode parameters
    var params = advanced ? get_form_values('#adv-form') : [];

    if (check_sources(index, panel_id)) { // all files are ok
	if (multiple) {
	    choose_function();
	} else {
	    $.post(
		routes['perform'],
		{ code   :  $('#sourcecode-'+index).text(),
		  lang   :  $('#lang-'+index).html(),
		  op     : operation,
		  adv    : advanced,
		  params : params,
		},
		function(data) {
		    $("#resultcode").html(data);
		    switch_buttons();
		}
	    );
	}
    }
    return true; //TODO
}
