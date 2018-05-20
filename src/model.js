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

dynamicGraphic();

async function solveEquation() {
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));
    const d = tf.variable(tf.scalar(Math.random()));

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

    function train(xs, ys, numIterations = 75) {
        const learningRate = 0.5;
        const optimizer = tf.train.sgd(learningRate);

        for (let iter = 0; iter < numIterations; iter++) {
            optimizer.minimize(() => {
                const predsYs = predict(xs);
                return loss(predsYs, ys);
            });
        }
    }
}
