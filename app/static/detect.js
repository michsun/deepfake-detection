function setNewSampleFile(text) {
    document.getElementById("dropdownMenuButton").textContent = text;
    document.getElementById("video-title").textContent = text;
    
    var static_sample_path = "static/media/sample-detected/"
    var new_detect_video_path = static_sample_path + "Detected-" + decodeURI(text);
    changeVideoSource(new_detect_video_path);
    
    var original_source = "app/static/media/sample-original/"+text;
    updateSampleResults(original_source);
};

function updateSampleResults(new_source) {
    pred = document.getElementById("df-prediction");
    conf = document.getElementById("df-confidence");
    
    pred.textContent = sample_results[new_source]['prediction'];
    var confidence = (sample_results[new_source]['prediction_confidence'] * 100).toFixed(2);
    conf.textContent = confidence+' %';

    document.getElementById("df-prediction").contentWindow.location.reload(true);
    document.getElementById("df-confidence").contentWindow.location.reload(true);
};

function changeVideoSource(new_source) {
    var video = document.getElementById('video');
    var sources = video.getElementsByTagName('source');
    var source = sources[0];
    video.pause();
    source.setAttribute('src', new_source);
    console.log({
        src: source.getAttribute('src'),
        type: source.getAttribute('type'),
    });
    video.load();
    video.play();
};

