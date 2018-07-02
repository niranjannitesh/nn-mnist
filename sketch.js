///<reference path="./libraries/p5.global-mode.d.ts" />

let mnist;
let train_index;

let train_img;

let test_index = 0;
let total_tests = 0;
let total_correct = 0;

let user_digit;
let user_has_drawing = false;

let user_guess_ele;
let percent_ele;

function setup() {
	createCanvas(400, 200).parent('container');
	background(0);

	user_digit = createGraphics(200, 200);
	user_digit.pixelDensity(1);

	train_img = createImage(28, 28);

	user_guess_ele = select('#guess');
	percent_ele = select('#percent');

	train_index = 0;
	mnist = loadAll();

	nnet = new Model(n_Error.meanSquaredError, 0.1);
	nnet.addLayer(new Layer(784));
	nnet.addLayer(new Layer(64, sigmoid));
	nnet.addLayer(new Layer(10, sigmoid));
}

function testing() {
	let inputs = [];
	for (let i = 0; i < 784; i++) {
		let bright = mnist.testing_imgs[i + test_index * 784];
		inputs[i] = bright / 255;
	}
	let label = mnist.testing_labels[test_index];

	let prediction = nnet.predict(inputs).toArray();
	let guess = prediction.indexOf(Math.max(...prediction));


	total_tests++;
	if (guess == label) {
		total_correct++;
	}

	let percent = 100 * (total_correct / total_tests);
	nnet.learning_rate = (100 - percent) / 150;
	percent_ele.html(nf(percent, 2, 2) + '%');


	test_index++;
	if (test_index == mnist.testing_labels.length) {
		test_index = 0;
		console.log('finished test set');
		console.log(percent);
		total_tests = 0;
		total_correct = 0;
	}
}

function guessUserDigit() {
	let img = user_digit.get();
	if (!user_has_drawing) {
		user_guess_ele.html('_');
		return img;
	}
	let inputs = [];
	img.resize(28, 28);
	img.loadPixels();
	for (let i = 0; i < 784; i++) {
		inputs[i] = img.pixels[i * 4] / 255;
	}
	let predict = nnet.predict(inputs).toArray();
	let guess = predict.indexOf(Math.max(...predict));
	user_guess_ele.html(guess);
	return img;
}

function train(show) {
	let inputs = [];
	// Image Display
	if (show) train_img.loadPixels();
	for (let i = 0; i < 784; i++) {
		let index = i * 4;
		let bright = mnist.traning_imgs[i + train_index * 784];
		inputs[i] = bright / 255;
		if (show) {
			train_img.pixels[index + 0] = bright;
			train_img.pixels[index + 1] = bright;
			train_img.pixels[index + 2] = bright;
			train_img.pixels[index + 3] = 255;
		}
	}
	if (show) {
		train_img.updatePixels();
		image(train_img, 200, 0, 200, height);
	}
	// Neural Network training
	let label = mnist.traning_labels[train_index];
	let targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	targets[label] = 1;

	// console.log(inputs);
	// console.log(targets);

	let predict = nnet.predict(inputs).toArray();
	let pred_label = predict.indexOf(Math.max(...predict));

	// console.log(pred_label);

	select('#lr').html(nnet.learning_rate.toFixed(4));

	nnet.train([inputs], [targets]);

	train_index = (train_index + 1) % mnist.traning_labels.length;
}


function draw() {
	if (mnist.traning_labels && mnist.testing_labels) {
		let user = guessUserDigit();
		// image(user, 0, 0);
		background(0);
		let total1 = 10;
		for (let i = 0; i < total1; i++) {
			if (i == total1 - 1) {
				train(true);
			} else {
				train(false);
			}
		}
		let total2 = 5;
		for (let i = 0; i < total2; i++) {
			testing();
		}
	}
	image(user_digit, 0, 0);

	if (mouseIsPressed) {
		user_has_drawing = true;
		user_digit.stroke(255, random(220, 255));
		user_digit.strokeWeight(18);
		user_digit.filter(BLUR, 1);
		user_digit.line(mouseX, mouseY, pmouseX, pmouseY);
	}
}


function keyPressed() {
	if (key == ' ') {
		user_has_drawing = false;
		user_digit.background(0);
	}
}
