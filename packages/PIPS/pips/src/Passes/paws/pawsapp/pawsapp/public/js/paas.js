function initialize_descriptions() {
    main_text();
    functionalities();
}		

function main_text() {
    $.ajax({
	type: "GET",
	cache: false,
	url: "/descriptions/paws_text",
	error: function() {
	    $('#left').html("Site can't be initialized!");
	},
	success: function(data){
	    $('#left').html('<h4>' + data + '</h4>');
	}
    });
}

function functionalities() {
    $.getJSON("/descriptions/sections", function(data){
	
	$.ajax({
	    type: "GET",
	    cache: false,
	    async: false,
	    url: "/descriptions/accordion",
	    error: function(){
		$('#right').html('ERROR');
	    },
	    success: function(acco) {
		$('#right').append(acco);
	    }
	});
	for (i=0; i<data.length; i++) {
	    $('#' + data[i]).accordion({
		active: false,
                animated: 'bounceslide',
		collapsible: true,
                autoHeight: false
	    });
	}
    });
}

$(function() {
    $("#basic_tools").accordion({ 
	active: false,
	animated: 'bounceslide',
	collapsible: true,
	autoHeight: false
    });

    $("#tutorial").accordion({ 
        active: false,
        animated: 'bounceslide',
        collapsible: true,
        autoHeight: false
    });

    $("#full_control").accordion({ 
        active: false,
        animated: 'bounceslide',
        collapsible: true,
        autoHeight: false
    });

    initialize_descriptions();
});
