var canvas = document.querySelector('#input_canvas');
var scaledCanvas = document.querySelector('#scaled_canvas');
var answer = document.querySelector('#answer');
var ctx = canvas.getContext('2d');
var scaled_ctx = scaledCanvas.getContext('2d');
var penColor = document.querySelector('input[name="penColor"]');
var penWidth = document.querySelector('input[name="penWidth"]');
var checker = document.querySelector('#checknumber');
var eraser = document.querySelector('#eraser');
ctx.strokeStyle = '#000000';
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 5;
scaled_ctx.strokeStyle = '#000000';
scaled_ctx.lineJoin = 'round';
scaled_ctx.lineCap = 'round';
scaled_ctx.lineWidth = 5;
var pen = {
    x: 0
    , y: 0
    , down: false
}
checker.addEventListener('click', submitImage);
eraser.addEventListener('click', clearImage);
canvas.addEventListener('mousedown', penDown);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', noDown);
canvas.addEventListener('mouseout', noDown);

function noDown() {
    pen.down = false;
}

function draw(e) {
    if (!pen.down) return;
    ctx.lineWidth = penWidth.value;
    ctx.strokeStyle = penColor.value;
    ctx.beginPath();
    ctx.moveTo(pen.x, pen.y);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    pen.x = e.offsetX;
    pen.y = e.offsetY;
}

function penDown(e) {
    pen.down = true;
    pen.x = e.offsetX;
    pen.y = e.offsetY;
}

function clearImage(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    scaled_ctx.clearRect(0, 0, scaledCanvas.width, scaledCanvas.height);
    txt = '';
    $("#answer").text(txt);
}

function submitImage() {
//    console.log(ctx.getImageData(0, 0, canvas.width, canvas.height));
    scaled_ctx.drawImage(canvas,0,0,scaledCanvas.width, scaledCanvas.height);
    var imageData = scaled_ctx.getImageData(0, 0, scaledCanvas.width, scaledCanvas.height);
//    console.log(imageData);
    var img_data = imageData.data;
    console.log(img_data);
    var len = img_data.length/4;
    var img_row = [];
    for (var i = 0; i < len; i += 1) {
      var alpha = 4*i+3;
      img_row.push(img_data[alpha]);
    }
    console.log(img_row);
    var csv = img_row.toString();
    $.post( "/guess_number", {
      image_str: csv
    }, function(data, status, resp){
      console.log(data);
      var text1 = 'It is '
      var text2 = ' ('
      var text3 ='%)'
      var display = text1.concat(data.answer, text2, data.percent,text3);
      console.log(display);
      $("#answer").text(display);
    });
}