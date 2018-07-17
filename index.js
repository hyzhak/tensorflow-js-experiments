"use strict";

import 'bootstrap/dist/css/bootstrap.css';

import * as main from './src/main';

document.addEventListener('DOMContentLoaded', () => {
  main.trainingPolynomialRegression({
    lossContainerId: 'loss',
    modelContainerId: 'model',
  });
});
