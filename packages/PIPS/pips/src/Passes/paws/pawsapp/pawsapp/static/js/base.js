/* skeleton.js */

var frameId = 'frame_ID';       // hidden frame id
var jFrame = null;              // hidden frame object
var formId = 'form_ID';         // hidden form id
var jForm = null;               // hidden form object
var files_no = 1;
var marker = '\", \"';
var multiple = false;
var file_name;

var directory;
var performed = false;
var created_graph = false;
var functions = '';

var options = {
    zoomHeight: 400,
    zoomWIDTH: 400
};

/*********************************************************************
*								     *
* Variables and functions for popup descriptions.		     *
*								     *
*********************************************************************/

var OffsetX = -10;
var OffsetY = 0;
var old, skn, iex = (document.all), yyy = -1000;
var ns4 = document.layers;
var ns6 = document.getElementById && !document.all;
var ie4 = document.all;

function popup(msg) {
    var content = '<div class="boxpopup">' + msg + "</div>";
    yyy = OffsetY;
    if (ns4) {skn.document.write(content); skn.document.close(); skn.visibility="visible";}
    if (ns6) {document.getElementById('pdqbox').innerHTML = content; skn.display = ""; skn.visibility="visible";}
    if (ie4) {document.all("pdqbox").innerHTML = content; skn.display = "";}
}

function get_mouse(e) {
    var x = (ns4 || ns6) ? e.pageX : event.x + document.body.scrollLeft;
    skn.left = x + OffsetX;
    var y = (ns4 || ns6) ? e.pageY : event.y + document.body.scrollTop;
    skn.top = y + yyy;
}

function remove_popup() {
    yyy = - 1000;
    if (ns4) {skn.visibility = "hidden";}
    else if (ns6 || ie4)
	skn.display = "none";
}

/*********************************************************************
*								     *
* Functions for user files uploading.				     *
*								     *
*********************************************************************/

function upload_prepare_forms() {
    jForm = createForm(formId);
    jFrame = createUploadIframe(frameId);

    jForm.appendTo('body');
    jForm.attr('target', frameId);

    $("#inp").bind('change', startUpload);
    function startUpload(){
	if(jForm!=null) {
	    jForm.remove();
	    jForm = createForm(formId);
	    jForm.appendTo('body');
	    jForm.attr('target', frameId);
	}

	var newElement = $(this).clone(true);
	newElement.bind('change',startUpload);
	newElement.insertAfter($(this));
	$(this).appendTo(jForm);

	jForm.submit();
	jFrame.unbind('load');
	jFrame.load(function() {
                upload_load(upload_get_response($(this)));
            });
    };
}

function upload_get_response(jframe) {
    var myFrame = document.getElementById(jframe.attr('name'));
    var bod = $(myFrame.contentWindow.document.body);
    var response = $(myFrame.contentWindow.document.body).text().substring(3, bod.text().length - 3);
    $('#pseudotextfile').value($('#inp').value());
    return response;
}

function createUploadIframe(id) {
    return $('<iframe width="300" height="200" name="' + id + '" id="' + id + '"></iframe>')
	.css({position: 'absolute', top: '270px', left: '450px', border:'1px solid #f00', display: 'none'})
	.appendTo('body');
};

function createForm(formId) {
    return $('<form method="post" action="/userfiles/upload" name="' + formId + '" id="' + formId +
	     '" enctype="multipart/form-data" style="position:absolute;width:10px;height:10px;left:45px;top:-45px;padding:5px;-moz-opacity:0;"></form>');
};

function split_response(response, sequence) {
    return response.replace(/\\n/g,'\n').replace(/\\t/g, '\t').split(sequence);
}

function set_response_to_first_tab(splited_0) {
    $("#source_tab_link1").text(splited_0[0]);
    $('#sourcecode1').val(splited_0[1]);
    language_detection(splited_0[1], 1);
}
               
function upload_add(response) {
    var splited = split_response(response, '\"], [\"');
    set_response_to_first_tab(splited[0].split(marker));
    change_tab_focus_src();
    files_no = splited.length;
    clear_multiple();
    if (files_no > 1) {
	upload_add_multiple(splited);
    }
};

function upload_load(response){
    removeTabs();
    clear_result_tab();
    upload_add(response);
    performed = false;
    created_graph = false;
}

function upload_add_multiple(splited) {
    for (var i = 1; i < splited.length; i++) {
	var no = i + 1;
	df = $('<div id="tabs-' + no + '"><form name="language' + no + '"><label for="lang' + no + '">Language: </label><input name="lang' + no + '" value="not yet detected." readonly="readonly" /></form><table><tr><td><form name="source' + no + '"><textarea name="sourcecode' + no + '" id="sourcecode' + no + '" rows="27" cols="120" onkeydown="handle_keydown(this, event)">' + splited[i].split(marker)[1] + '</textarea></form></td></tr></table></div>');
	df.appendTo('#tabs');
	$('#sourcecode' + no).attr('spellcheck', false);
	$('#tabs').tabs('add', '#tabs-' + no, splited[i].split(marker)[0], i);
	language_detection(splited[i].split(marker)[1], no);
    }
    multiple = true;
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
		$('#lang' + number).val(data);
                test = true;
	    } else {
		$('#lang' + number).val(" not supported.");
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
        data: { code: source, language: $('#lang' + number).val() },
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
    var sourcee  = $('#sourcecode' + selected).val();
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
		$('save-button').attr('href', '/res/' + directory + '/' + directory);
	}
    });
}
 	
function get_functions() {

    var datas = {};
    datas['number'] = files_no;
    for(var i=0; i<files_no; i++) {
	datas['code' + i] = $('#sourcecode' + (i+1)).text();
	datas['lang' + i] = $('#lang' + (i+1)).val();
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

    var sources   = new Array(files_no),
	languages = new Array(files_no);

    get_directory(panel_id);
    
    for (var i = 1; i <= files_no; i++) {
	sources[i] = $('#sourcecode' + i).text();
    }
    var u = 1;
    while ((u <= files_no) && preprocess_input(sources[u], index, u, panel_id)) {
	u = u + 1;
    }
    if (u == files_no + 1) {
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
		{ code:     $('#sourcecode1').text(),
                  language: $('#lang1').val()
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
* Helper functions for keeping consistency between input and output. *
*								     *
*********************************************************************/

function activate_tab(id) {

    $link = $('#' + id, '#op-tabs');

    // Activate selected tab
    $('li', '#op-tabs').removeClass('active');
    $link.parent().addClass('active');

    // Activate corresponding panel
    $('.tab-pane', '#op-tabs').removeClass('active');
    $($link.attr('href')).addClass('active');
}

function add_tab_title(text) {
    $("#result_tab_link").text(text);
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

function activate_buttons() {
    $("#save-button").removeClass("disabled");
    $("#print-button").removeClass("disabled");
}

function deactivate_buttons() {
    $("#save-button").addClass("disabled");
    $("#print-button").addClass("disabled");
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

function clear_multiple() {
    multiple = false;
    $('#multiple-functions').html('');
}

function removeTabs() {
    for(var i = 1; i < files_no; i++) {
	$('#tabs').tabs('remove', 1);
    }
    files_no = 1;
    multiple = false;
}


/*********************************************************************
*								     *
* Functions for loading examples.				     *
*								     *
*********************************************************************/


function load_example(filename) {

    $('#classic-examples-dialog').modal('hide');
    if (multiple)
	removeTabs();
    $("#source_tab_link1").text(filename);

    var url = routes['get_example_file'].replace('{tool}', operation).replace('{filename}', filename);
    $.get(url, function(data) {
        $('#sourcecode1').text(data);
        $('#pseudotextfile').val('');
        activate_tab('source_tab_link1');
        deactivate_buttons();
	performed     = false;
	created_graph = false;
	clear_result_tab();
	clear_multiple();
	language_detection(data, 1);
    });
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
		    language: $('#lang1').val()
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
		{ code:      $('#sourcecode'+index).text(),
		  language:  $('#lang'+index).val(),
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
