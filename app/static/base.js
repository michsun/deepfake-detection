// Autoremove alerts
window.onload = function(){
    setTimeout(function(){ 
        $(".alert").removeClass("show");
        $(".alert").addClass("hide");
        setTimeout(function() {
            $(".alert-content").remove();
        }, 850);
    }, 8000);
};

jQuery(document).ready(function( $ ){

    $(".close-btn").click(function(){
        $(this).parent().removeClass("show");
        $(this).parent().addClass("hide");
        $alert = $(this);
        setTimeout(function() {
            $alert.parent().remove();
        }, 850);
    });

});

