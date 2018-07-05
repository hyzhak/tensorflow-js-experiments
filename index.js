"use strict";

import * as model from './src/model';

document.addEventListener('DOMContentLoaded', () => {
  model.trainingPolynomialRegression({lossGraphicContainerId: 'graphics'});
});
