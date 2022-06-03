// $(document).ready(function(){
//     var filename = $(".dropdown-item").click(function(){
//         document.getElementById("dropdown-videos").text = filename;
//         console.log(filename);
//     }).text();
    
// });


// var video = document.getElementById('video');
// var source = video.getElementByTagName('source');
// source.setAttribute('src', "{{ url_for('static', filename='sample-media/zelensky-deepfake.mp4') }}");
// source.setAttribute('type', 'video/ogg');

function setNewFile(text) {
    document.getElementById("dropdown-videos").textContent = text;
    
    document.getElementById("video-title").textContent = text;
    
    var static_sample_path = "static/media/sample-detected/"
    var 
    
    new_detect_video_path = static_sample_path + "Detected-" + decodeURI(text);
    // console.log(new_sample_video_path);
    changeVideoSource(new_detect_video_path);
    
    var original_source = "app/static/media/sample-originals/"+text;
    updateResults(original_source);
};

function updateResults(new_source) {
    pred = document.getElementById("df-prediction");
    conf = document.getElementById("df-confidence");

    pred.textContent = '{{ results["' + new_source + '"]["prediction"] }}';
    conf.textContent = '{{ results["' + new_source + '"]["prediction_confidence"] }}';

    // pred.textContent = ;
    // pred.text = {{ results['"'+new_source+'"']['prediction']}};
    

    document.getElementById("df-prediction").contentWindow.location.reload(true);
    document.getElementById("df-confidence").contentWindow.location.reload(true);
    // document.reload(true);
}

function changeVideoSource(new_source) {
    var video = document.getElementById('video');
    var sources = video.getElementsByTagName('source');
    var source = sources[0];
    video.pause();
    source.setAttribute('src', new_source);
    // source.setAttribute('src', Flask.url_for('static', new_source));
    console.log({
        src: source.getAttribute('src'),
        type: source.getAttribute('type'),
    });
    video.load();
    video.play();
};




// video.appendChild(source);
// video.play();
// console.log({
//   src: source.getAttribute('src'),
//   type: source.getAttribute('type'),
// });

// setTimeout(function() {
//   video.pause();

//   source.setAttribute('src', 'http://techslides.com/demos/sample-videos/small.webm');
//   source.setAttribute('type', 'video/webm');

//   video.load();
//   video.play();
//   console.log({
//     src: source.getAttribute('src'),
//     type: source.getAttribute('type'),
//   });
// }, 3000);



// DROPDOWN MENU


// const sample_dir = 'app/static/sample-media/'
// var fs = require('fs');
// var sample_files = fs.readdirSync(sample_dir);
// // console.log(files)
// function iterateArray(value, index, array) {
//     return value;
// }
// let get_sample_files = sample_files.iterateArray(files)
// document.getElementById("sample-dropdown").innerHTML = `
// `;

// function initializeFileUploads() {
//     $('.file-upload').change(function () {
//         var file = $(this).val();
//         $(this).closest('.input-group').find('.file-upload-text').val(file);
//     });
//     $('.file-upload-btn').click(function () {
//         $(this).find('.file-upload').trigger('click');
//     });
//     $('.file-upload').click(function (e) {
//         e.stopPropagation();
//     });
// }


// // On document load:
// $(function() {
//     initializeFileUploads();
// });

