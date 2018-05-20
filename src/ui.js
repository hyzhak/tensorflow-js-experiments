/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import renderChart from 'vega-embed';

export async function plotData(container, xs, ys) {
    const xvals = xs.data ? await xs.data() : xs;
    const yvals = ys.data ? await ys.data() : ys;

    const values = Array.from(yvals).map((y, i) => {
        return {'x': xvals[i], 'y': yvals[i]};
    });

    const spec1 = {
        // "$schema": "https://vega.github.io/schema/vega/v4.json",
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'width': 300,
        'height': 300,
        'data': {name: 'table', values},
        'mark': 'point',
        // 'mark': 'line',
        'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y': {'field': 'y', 'type': 'quantitative'},
        },
    };

    const spec2 = {
        "$schema": "https://vega.github.io/schema/vega/v4.json",
        "width": 400,
        "height": 200,
        "padding": 5,

        "data": [
            {
                "name": "table",
                "values": [
                    {"category": "A", "amount": 28},
                    {"category": "B", "amount": 55},
                    {"category": "C", "amount": 43},
                    {"category": "D", "amount": 91},
                    {"category": "E", "amount": 81},
                    {"category": "F", "amount": 53},
                    {"category": "G", "amount": 19},
                    {"category": "H", "amount": 87}
                ]
            }
        ],

        "signals": [
            {
                "name": "tooltip",
                "value": {},
                "on": [
                    {"events": "rect:mouseover", "update": "datum"},
                    {"events": "rect:mouseout", "update": "{}"}
                ]
            }
        ],

        "scales": [
            {
                "name": "xscale",
                "type": "band",
                "domain": {"data": "table", "field": "category"},
                "range": "width",
                "padding": 0.05,
                "round": true
            },
            {
                "name": "yscale",
                "domain": {"data": "table", "field": "amount"},
                "nice": true,
                "range": "height"
            }
        ],

        "axes": [
            {"orient": "bottom", "scale": "xscale"},
            {"orient": "left", "scale": "yscale"}
        ],

        "marks": [
            {
                "type": "rect",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"scale": "xscale", "field": "category"},
                        "width": {"scale": "xscale", "band": 1},
                        "y": {"scale": "yscale", "field": "amount"},
                        "y2": {"scale": "yscale", "value": 0}
                    },
                    "update": {
                        "fill": {"value": "steelblue"}
                    },
                    "hover": {
                        "fill": {"value": "red"}
                    }
                }
            },
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "align": {"value": "center"},
                        "baseline": {"value": "bottom"},
                        "fill": {"value": "#333"}
                    },
                    "update": {
                        "x": {"scale": "xscale", "signal": "tooltip.category", "band": 0.5},
                        "y": {"scale": "yscale", "signal": "tooltip.amount", "offset": -2},
                        "text": {"signal": "tooltip.amount"},
                        "fillOpacity": [
                            {"test": "datum === tooltip", "value": 0},
                            {"value": 1}
                        ]
                    }
                }
            }
        ]
    };

    return renderChart(container, spec2, {actions: false});
}

export async function plotDataAndPredictions(container, xs, ys, preds) {
    const xvals = await xs.data();
    const yvals = await ys.data();
    const predVals = await preds.data();

    const values = Array.from(yvals).map((y, i) => {
        return {'x': xvals[i], 'y': yvals[i], pred: predVals[i]};
    });

    const spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'width': 300,
        'height': 300,
        'data': {'values': values},
        'layer': [
            {
                'mark': 'point',
                'encoding': {
                    'x': {'field': 'x', 'type': 'quantitative'},
                    'y': {'field': 'y', 'type': 'quantitative'}
                }
            },
            {
                'mark': 'line',
                'encoding': {
                    'x': {'field': 'x', 'type': 'quantitative'},
                    'y': {'field': 'pred', 'type': 'quantitative'},
                    'color': {'value': 'tomato'}
                },
            }
        ]
    };

    return renderChart(container, spec, {actions: false});
}
