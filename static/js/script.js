// Initialize socket.io for real-time communication
var socket = io();

// Get DOM elements
var startRecordingButton = document.getElementById('start-recording');
var stopRecordingButton = document.getElementById('stop-recording');
var audioContainer = document.getElementById('audio-container');

// Initialize MediaRecorder
var mediaRecorder;
var chunks = [];

// Event listener for start recording button
startRecordingButton.addEventListener('click', function() {
    startRecordingButton.disabled = true;
    stopRecordingButton.disabled = false;

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = function(event) {
                chunks.push(event.data);
            }

            mediaRecorder.onstop = function() {
                var audioBlob = new Blob(chunks, { type: 'audio/wav' });
                var audioUrl = URL.createObjectURL(audioBlob);
                var audio = new Audio(audioUrl);
                audio.controls = true;
                audioContainer.innerHTML = '';
                audioContainer.appendChild(audio);

                // Send audio data to server (you'll need to implement this)
                socket.emit('audio', audioBlob);
            }

            mediaRecorder.start();
        })
        .catch(function(error) {
            console.error('Error accessing microphone:', error);
        });
});

// Event listener for stop recording button
stopRecordingButton.addEventListener('click', function() {
    startRecordingButton.disabled = false;
    stopRecordingButton.disabled = true;

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
});
