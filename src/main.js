import './main.scss';

import {zip} from 'lodash';
import * as tf from '@tensorflow/tfjs';
import * as vega from 'vega';
import vegaEmbed from 'vega-embed';

import {polynomialRegression} from './models/polynomial-regression';


const lossGraphicSpec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
  "title": "loss",
  "data": {
    "name": "loss",
  },
  "width": 400,
  "mark": "line",
  "encoding": {
    "x": {
      "field": "x",
      "type": "quantitative",
      "scale": {"domain": [0.0, 80.0]},
      "axis": {"title": "iteration"},
    },
    "y": {
      "field": "y",
      "type": "quantitative",
      "scale": {"domain": [0.0, 0.25]},
      "axis": {"title": "loss"},
    },
  }
};

const modelGraphicSpec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
  "width": 400,
  "title": "model",
  "layer": [{
    "data": {
      "name": "dataset",
    },
    "mark": "circle",
    "encoding": {
      "x": {"field": "x", "type": "quantitative"},
      "y": {"field": "y", "type": "quantitative", "scale": {"domain": [0.0, 1.0]}},
      "title": "dataset",
    }
  }, {
    "data": {
      "name": "approximation",
    },
    "mark": "line",
    "encoding": {
      "x": {"field": "x", "type": "quantitative"},
      "y": {"field": "y", "type": "quantitative", "scale": {"domain": [0.0, 1.0]}},
      "title": "approximation",
    }
  }],
};

function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b),
      tf.scalar(coeff.c), tf.scalar(coeff.d)
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
  });
}

/**
 * The pipeline of training polynomial regression model
 *
 * - create the model
 * - create loss functions graphic
 * - train the model and draw the loss function
 * - draw the model (on-fly)
 *
 * @param lossContainerId
 * @param modelContainerId
 * @returns {Promise.<void>}
 */
export async function trainingPolynomialRegression({lossContainerId, modelContainerId}) {
  const learningRate = 0.5;
  const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
  const trainingData = generateData(100, trueCoefficients);

  const model = polynomialRegression();

  // draw graphic
  const lossContainerEl = document.getElementById(lossContainerId);
  if (!lossContainerEl) {
    throw new Error(`Loss container ${lossContainerId} is not defined`);
  }

  const lossGraphics = await vegaEmbed(lossContainerEl, lossGraphicSpec, {
    actions: false,
  });

  // track loss function changes and reflec them to loss graphics
  let index = 0;
  model.lossStream
    .subscribe(y => {
      lossGraphics.view.change(
        'loss',
        vega.changeset().insert([{x: index, y}]),
      ).run();
      index++;
    });

  const modelContainerEl = document.getElementById(modelContainerId);
  if (!modelContainerEl) {
    throw new Error(`Model container ${modelContainerId} is not defined`);
  }

  const xs = await trainingData.xs.data();
  const xy = await trainingData.ys.data();

  const dataset = zip(xs, xy).map(([x, y]) => ({x, y}));
  modelGraphicSpec.layer[0].data.values = dataset;
  const modelGraphics = await vegaEmbed(modelContainerEl, modelGraphicSpec, {
    actions: false,
  });
  model.predictionStream
    .subscribe(async function (a, b, c, d) {
      // draw prediction

      // 1) generate points (x,y)
      let points = await model.modelValues();
      points = zip(points.x, points.y).map(([x, y]) => ({x, y}));

      // 2) add data to
      modelGraphics.view.change(
        'approximation',
        vega.changeset().remove(() => true).insert(points),
      ).run();
    });

  await model.train({numIterations: 75, trainingData});
}
