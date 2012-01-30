// Global vars
var directory,
    performed = false,
    created_graph = false,
    functions = '',
    nb_files = 1,
    multiple = false,
    
    options = {
	zoomHeight: 400,
	zoomWIDTH: 400
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
    $('.tab-pane', '#op-tabs').removeClass('active');
    $($link.attr('href')).addClass('active');
}

function change_tab_focus_res(result_tab_index, panel_id) {
    $('#tabs').tabs("select", result_tab_index);
    add_wait_notification(panel_id);
}

function add_wait_notification(panel_id) {
    var tab = '#resultcode';
    if (panel_id == 'graph')
	tab = '#graph';
    $(tab).html('<div class="alert-message info"><b>Please wait while processing...</b> It might take a long time.</div>');
}

function add_choose_function_notification() {
    $('#resultcode').html("<p><b>Choose function to display.</b><br/></p>");
}

function enable(sel) {
    $(sel).removeClass("disabled");
}
function disable(sel) {
    $(sel).addClass("disabled");
}

function activate_buttons() {
    disable('#run-button');
    enable('#save-button');
    enable('#print-button');
}
function deactivate_buttons() {
    enable('#run-button');
    disable('#save-button');
    disable('#print-button');
}

function deactivate_graph_buttons() {
    //console.debug('DEACTIVATING');
}

function activate_graph_buttons() {
    //console.debug('ACTIVATING');
}

function clear_result_tab() {
    deactivate_buttons();
    $('#resultcode').html('');
    // all of other tabs are removed
    $('#lang1').val('not yet detected');
}

function clear_graph_tab() {
    deactivate_graph_buttons();
    add_wait_notification('graph');
}

// Delete extra source tabs and panels
function clear_multiple() {
    activate_tab('source-1');
    var tabs = $('#op-tabs ul.tabs li');
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
    clear_result_tab();

    nb_files = files.length;

    for (var i=1; i<= nb_files; i++) {

	var fname = files[i-1][0],
	    code  = files[i-1][1];

	// Create tab and panel on the fly
	if (i > 1) {
	    $("#source-" + (i-1) + "_tab").after('<li id="source-' + i + '_tab"><a href="#source-' + i + '">SOURCE</a></li>');
	    $('#source-' + (i-1)).after($('#source_panel_skel').html().replace(/__skel__/g, i));
	    $('#sourcecode-' + i).attr('spellcheck', false);
	}
	$("#source-" + i + "_tab a").text(fname);
	$('#sourcecode-' + i).text(code);
	language_detection(code, i);
    }
    if (nb_files > 1)
	for (var i=2; i<= nb_files; i++) {
	    //$('#sourcecode-' + i).linedtextarea();
	};

    deactivate_buttons();
    multiple      = (nb_files > 1);
    performed     = false;
    created_graph = false;
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
            if (data != "none") {
		$('#lang-' + number).val(data);
                test = true;
	    } else {
		$('#lang-' + number).val(" not supported.");
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
        data: { code: source, language: $('#lang-' + number).val() },
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
	datas['lang' + i] = $('#lang-' + (i+1)).val();
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
			activate_buttons();
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
		{ code:     $('#sourcecode-1').text(),
                  language: $('#lang-1').val()
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
    clear_result_tab();
    enable('#run-button');
    performed = false;
    created_graph = false;
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
    changeFontSize('body', direction);
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
		    language: $('#lang-1').val()
		  },
	    cache: false,
	    url: routes['perform_multiple'],
	    error: function() {},
	    success: function(data) {
		$('#resultcode').html(data);
		activate_buttons();
	    }
	});
    });
    add_choose_function_notification();
}
		

function perform_operation(index, panel_id) {
    clear_result_tab();
    if (check_sources(index, panel_id)) { // all files are ok
	if (multiple) {
	    choose_function();
	} else {
	    $.post(
		routes['perform'],
		{ code:      $('#sourcecode-'+index).text(),
		  language:  $('#lang-'+index).val(),
		  operation: operation
		},
		function(data) {
		    $("#resultcode").html(data);
		    activate_buttons();
		}
	    );
	}
    }
    return true; //TODO
}
