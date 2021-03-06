import * as tf from '@tensorflow/tfjs';
import {Observable} from 'rxjs';

function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

class PolynomialRegressionModel {
  constructor(learningRate = 0.5) {
    this.a = tf.variable(tf.scalar(Math.random()));
    this.b = tf.variable(tf.scalar(Math.random()));
    this.c = tf.variable(tf.scalar(Math.random()));
    this.d = tf.variable(tf.scalar(Math.random()));

    this.optimizer = tf.train.sgd(learningRate);

    this.lossStream = new Observable.create((observer) => {
      this.lossObserver = observer;
    });

    this.predictionStream = new Observable.create((observer) => {
      this.predictionObserver = observer;
    });
  }

  /**
   * y = a * x ^ 3 + b * x ^ 2 + c * x + d
   * @param x
   */
  model(x) {
    return this.a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(this.b.mul(x.square())) // + b * x ^ 2
      .add(this.c.mul(x)) // + c * x
      .add(this.d); // + d
  }

  async modelValues() {
    const xspace = tf.linspace(-1, 1, 100);
    const x = xspace.dataSync();
    const y = await this.model(xspace).data();
    return {x, y};
  }

  predict(x) {
    return tf.tidy(() => this.model(x));
  }

  async train(xs, ys, numIterations = 75) {
    for (let iter = 0; iter < numIterations; iter++) {
      this.optimizer.minimize(() => {
        const predsYs = this.predict(xs);
        const lossRes = loss(predsYs, ys);
        lossRes.data()
          .then((value) => {
            this.lossObserver.next(value[0]);
          });
        this.predictionObserver.next();
        return lossRes;
      });

      // Use tf.nextFrame to not block the browser.
      await tf.nextFrame();
    }

    this.lossObserver.complete();
    this.predictionObserver.complete();
  }
}

export function polynomialRegression() {
  const model = new PolynomialRegressionModel();

  return {
    lossStream: model.lossStream,
    predictionStream: model.predictionStream,
    modelValues: model.modelValues.bind(model),
    train: async function ({numIterations, trainingData}) {
      // estimate before training
      const predictionsBefore = model.predict(trainingData.xs);
      predictionsBefore.sub(trainingData.ys).mean().print();

      //train
      await model.train(trainingData.xs, trainingData.ys, numIterations)

      // estimate after training
      const predictionsAfter = model.predict(trainingData.xs);
      predictionsAfter.sub(trainingData.ys).mean().print();

      predictionsBefore.dispose();
      predictionsAfter.dispose();
    }
  };
}
