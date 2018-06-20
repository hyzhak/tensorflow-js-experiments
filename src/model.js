import * as tf from '@tensorflow/tfjs';

import * as vega from 'vega';
import vegaEmbed from 'vega-embed';

import {polynomialRegression} from './models/polynomial-regression';
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

const lossGraphicSpec = {
  '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
  'data': {
    'name': "loss",
  },
  'width': 400,
  'mark': 'line',
  'encoding': {
    'x': {'field': 'x', 'type': 'quantitative'},
    'y': {"field": "y", "type": "quantitative"},
  }
};

/**
 * The pipeline of training polynomial regression model
 *
 * - create the model
 * - create loss functions graphic
 * - train the model and draw the loss function
 *
 * TODO:
 * - draw the model (on-fly)
 *
 * @param lossGraphicContainerId
 * @returns {Promise.<void>}
 */
async function trainingPolynomialRegression({lossGraphicContainerId}) {
  const model = polynomialRegression();

  // draw graphic
  const parentEl = document.getElementById(lossGraphicContainerId);
  console.log('parentEl', parentEl);
  if (!parentEl) {
    throw new Error(`target element ${lossGraphicContainerId} is not defined`)
  }

  const result = await vegaEmbed(parentEl, lossGraphicSpec);

  let index = 0;
  model.lossStream
    .subscribe(y => {
      result.view.change(
        'loss',
        vega.changeset().insert([{x: index, y}]),
      ).run();
      index++;
    });

  await model.train();
}

setTimeout(() => {
  trainingPolynomialRegression({lossGraphicContainerId: 'graphics'});
});
