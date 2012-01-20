$(function(){
    initialize();
});

$(document).ready(function() {
    upload_prepare_forms();
});

function choose_function() {
    $('#multiple-functions').html(functions);
    $('#multiple-functions input:submit').button();
    $('#multiple-functions input:submit').click(function() {
	$.ajax({
	    type: "POST",
	    data: {
		function: $(this).attr('value'),
		operation: ${self.operation_id()},
		language: document.language1.lang1.value
	    },
	    cache: false,
	    url: "/operations/perform_multiple",
	    error: function() {
	    },
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
	    invoke_operation(document.source1.sourcecode1.value, document.language1.lang1.value);
	}
    }
}

function invoke_operation(source_code, lang) {
    $.ajax({
	type: "POST",
	data: {
	    code: source_code,
	    language: lang,
	    operation: ${self.operation_id()}
	},
	cache: false,
	url: "/operations/perform", 
	error: function() {
	    $("#resultcode").html("Web error, try again!");
	},
	success: function(data) {
	    $("#resultcode").html(data);
	    activate_buttons();
	}
    });
}
