import vegaEmbed from 'vega-embed';


/**
 * Build model graphics
 *
 * @param containerId
 * @param values
 * @returns {Promise.<*>}
 */
export async function build(containerId, values) {
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
        "values": values,
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
      "mark": {
        "color": "red",
        "type": "line",
      },
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
