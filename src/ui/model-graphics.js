import vegaEmbed from 'vega-embed';

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


/**
 * Build model graphics
 *
 * @param containerId
 * @returns {Promise.<*>}
 */
export async function build(containerId, dataSetValues) {
  const containerEl = document.getElementById(containerId);
  if (!containerEl) {
    throw new Error(`Model container ${containerId} is not defined`);
  }
  return await vegaEmbed(containerEl, {
    "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
    "width": 400,
    "title": "model",
    "layer": [{
      "data": {
        "name": "dataset",
        "values": dataSetValues,
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
  }, {
    actions: false,
  });
}
