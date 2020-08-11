
function renderVis(data, id) {
    var yourVlSpec = {
      "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
      "data": {"values": data},
      "title": "MF",
      "mark": "rect",
      "encoding": {
        "x": {
            "field": "items",
            "type": "ordinal",
            "title": "movies",
            "axis": {
                "labelAngle": 0
            }
        },
        "y": {
            "field": "users",
            "type": "ordinal",
            "title": "users"
        },
        "color": {
            "field": "ratings",
            "type": "quantitative",
            "legend": {
                "title": "Ratings"
          }
        }
      }
    }
    vegaEmbed(id, yourVlSpec);
  }

  function strip(number) {
    return (parseFloat(number).toPrecision(5));
  }

  function renderEmbedding(embeddingArray, id) {

    const embedding = embeddingArray.map(function (d, i){
      return {
        id: i + 1,
        values: d
      }
    })
    
    let yourVlSpec = {
      "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
      "data": {"values": embedding},
      "title": "UserEmbedding",
      "transform": [
        {"flatten": ["values"]},
        {
          "window": [{"op": "row_number", "as": "idx" }],
          "groupby": ["id"]
        }
      ],
      "mark": "rect",
      "encoding": {
        "x": {
            "field": "id",
            "type": "ordinal",
            "axis": {
                "labelAngle": 0
            }
        },
        "y": {
            "field": "idx",
            "type": "ordinal",
        }
        ,
        "color": {
            "field": "values",
            "type": "quantitative",
            "legend": {
                "title": null
          }
        }
      }
    }
    vegaEmbed(id, yourVlSpec);
  }

export {renderVis, renderEmbedding, strip};