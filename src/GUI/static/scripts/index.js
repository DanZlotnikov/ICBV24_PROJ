function onImageLoad() {
    scrollToEnhanceSection()
    var image = document.getElementById('uploadedImage');
    var imagePath = image.getAttribute('src'); // This gets the full URL of the image
    var imageName = imagePath.split('/').pop(); // This splits the path by '/' and gets the last segment

    var canvas = document.getElementById('imageCanvas');
    var ctx = canvas.getContext('2d');

    // Set canvas dimensions to match the image
    canvas.width = image.width;
    canvas.height = image.height;

    // Position the canvas over the image
    canvas.style.position = 'absolute';
    canvas.style.left = image.offsetLeft + 'px';
    canvas.style.top = image.offsetTop + 'px';

    var canvas = document.getElementById('imageCanvas');
    var ctx = canvas.getContext('2d');
    var rect = {};
    var drag = false;

    function init() {
        canvas.addEventListener('mousedown', mouseDown, false);
        canvas.addEventListener('mouseup', mouseUp, false);
        canvas.addEventListener('mousemove', mouseMove, false);
    }

    function mouseUp() {
        drag = false;
        // Now we have the rectangle coordinates, you can send them to the server
        // This is an example, you need to implement the function to do the actual sending
        sendRectToServer(rect);
    }

    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    function mouseDown(e) {
        var pos = getMousePos(canvas, e);
        rect.startX = pos.x;
        rect.startY = pos.y;
        rect.Name = imageName;
        drag = true;
    }

    function mouseMove(e) {
        if (drag) {
            var pos = getMousePos(canvas, e);
            rect.w = pos.x - rect.startX;
            rect.h = pos.y - rect.startY;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.setLineDash([6]);
            ctx.strokeStyle = '#4CAF50'
            ctx.fillStyle = 'rgba(76, 175, 80, 0.5)'; // semi-transparent green
            ctx.fillRect(rect.startX, rect.startY, rect.w, rect.h); // Fill the rectangle with color

            ctx.strokeRect(rect.startX, rect.startY, rect.w, rect.h);
        }
    }


    function sendRectToServer(rect) {
        // Example AJAX call to send rectangle coordinates to the server
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/process-rect", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.send(JSON.stringify(rect));
    }

    init();
}


function scrollToEnhanceSection() {
    // Get the element you want to scroll to
    var element = document.getElementById("enhanceImageSection");
    // Use scrollIntoView to scroll to the element
    element.scrollIntoView({ behavior: 'smooth' });
}


window.onload = function() {
// Set the hidden input's value to the image source
var imageSrc = document.getElementById('uploadedImage').src;
document.getElementById('imageSrc').value = imageSrc;
};
