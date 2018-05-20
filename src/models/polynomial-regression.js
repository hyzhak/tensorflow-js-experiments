import * as tf from '@tensorflow/tfjs';

export async function solvePolynomialRegression() {
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