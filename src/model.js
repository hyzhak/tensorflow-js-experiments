import * as tf from '@tensorflow/tfjs';

import * as vega from 'vega';
import vegaEmbed from 'vega-embed';

import * as ui from './ui';

async function singleUnitExample() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    // Train the model using the data.
    await model.fit(xs, ys);

    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print();
}

// singleUnitExample();

async function dynamicGraphic() {
    // wait until DOM will be rendered
    await new Promise((resolve) => setTimeout(resolve, 100));

    // draw graphic
    const parentEl = document.getElementById('graphics');

    var vlSpec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'name': 'table'},
        'width': 400,
        'mark': 'line',
        'encoding': {
            'x': {'field': 'x', 'type': 'quantitative', 'scale': {'zero': false}},
            'y': {'field': 'y', 'type': 'quantitative'},
            'color': {'field': 'category', 'type': 'nominal'}
        }
    };

    const res = await vegaEmbed(parentEl, vlSpec, {
        actions: false,
    });

    /**
     * Generates a new tuple with random walk.
     */
    function seriesGenerator() {
        var counter = -1;
        var previousY = [0, 0, 0, 0];
        return () => {
            counter++;
            var newVals = previousY.map((v, c) => ({
                x: counter,
                y: v + Math.random() * c - c / 2,
                category: c
            }));
            previousY = newVals.map((v) => v.y);
            return newVals;
        };
    }

    // generate data on-fly
    var valueGenerator = seriesGenerator();
    var minimumX = -100;

    window.setInterval(() => {
        minimumX++;

        const changeSet = vega.changeset()
            .insert(valueGenerator())
            .remove((t) => t.x < minimumX);

        res.view
            .change('table', changeSet)
            .run();
    }, 1000 / 30);
}

// dynamicGraphic();

async function solvePolynomialEquation() {
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));
    const d = tf.variable(tf.scalar(Math.random()));

    function generateData(numPoints, coeff, sigma = 0.04) {
        return tf.tidy(() => {
            const [a, b, c, d] = [
                tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
                tf.scalar(coeff.d)
            ];

            const xs = tf.randomUniform([numPoints], -1, 1);

            // Generate polynomial data
            const three = tf.scalar(3, 'int32');
            const ys = a.mul(xs.pow(three))
                .add(b.mul(xs.square()))
                .add(c.mul(xs))
                .add(d)
                // Add random noise to the generated data
                // to make the problem a bit more interesting
                .add(tf.randomNormal([numPoints], 0, sigma));

            // Normalize the y values to the range 0 to 1.
            const ymin = ys.min();
            const ymax = ys.max();
            const yrange = ymax.sub(ymin);
            const ysNormalized = ys.sub(ymin).div(yrange);

            return {
                xs,
                ys: ysNormalized
            };
        })
    }

    function predict(x) {
        // y = a * x ^ 3 + b * x ^ 2 + c * x + d
        return tf.tidy(() => {
            return a.mul(x.pow(tf.scalar(3))) // a * x^3
                .add(b.mul(x.square())) // + b * x ^ 2
                .add(c.mul(x)) // + c * x
                .add(d); // + d
        });
    }

    function loss(predictions, labels) {
        // Subtract our labels (actual values) from predictions, square the results,
        // and take the mean.
        const meanSquareError = predictions.sub(labels).square().mean();
        return meanSquareError;
    }

    async function train(xs, ys, numIterations = 75) {
        const optimizer = tf.train.sgd(learningRate);

        for (let iter = 0; iter < numIterations; iter++) {
            optimizer.minimize(() => {
                const predsYs = predict(xs);
                return loss(predsYs, ys);
            });

            // Use tf.nextFrame to not block the browser.
            await tf.nextFrame();
        }
    }

    const learningRate = 0.5;
    const numIterations = 75;
    const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
    const trainingData = generateData(100, trueCoefficients);

    const predictionsBefore = predict(trainingData.xs);
    predictionsBefore.sub(trainingData.ys).mean().print();

    // Train the model!
    await train(trainingData.xs, trainingData.ys, numIterations);

    const predictionsAfter = predict(trainingData.xs);
    predictionsAfter.sub(trainingData.ys).mean().print();


    predictionsBefore.dispose();
    predictionsAfter.dispose();
}

solvePolynomialEquation();
