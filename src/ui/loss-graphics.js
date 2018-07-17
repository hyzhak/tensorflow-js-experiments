import vegaEmbed from 'vega-embed';


const lossGraphicSpec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
  "title": "loss",
  "data": {
    "name": "loss",
  },
  "width": 400,
  "mark": {
    "color": "red",
    "type": "line",
  },
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

/**
 * Build loss graphics
 *
 * @param containerId
 * @returns {Promise.<*>}
 */
export async function build(containerId) {
  // draw graphic
  const containerEl = document.getElementById(containerId);
  if (!containerEl) {
    throw new Error(`Loss container ${containerId} is not defined`);
  }

  return await vegaEmbed(containerEl, lossGraphicSpec, {
    actions: false,
  });
}
