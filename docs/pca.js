console.log('Hello Script');


function renderVis(data, id) {
  var yourVlSpec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
    "data": {"values": data},
    "title": "PCA",
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
              "title": null
        }
      }
    }
  }
  vegaEmbed(id, yourVlSpec);
}



/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    console.log("getting data...")
    const rawFile = await fetch('sample/ratingsU.json');  
    const rawJSON = await rawFile.json();  
    console.log("loaded data...")
    const data = rawJSON.map(d => ({
      users: d.users,
      items: d.movies,
      ratings: d.ratings
    }));
    return data;
  }

  function createModel(nItems, nUsers, nFactors) {

    // Item Layer
    const itemInput = tf.layers.input({
      shape: [1], 
      name: 'Item'
    })

    const itemEmbedding = tf.layers.embedding({
      inputDim: nItems, 
      outputDim: nFactors,
      name: 'ItemEmbedding'
    }).apply(itemInput)

    const itemVec = tf.layers.flatten({
      name: "ItemFlat"
    }).apply(itemEmbedding)

    // User Layer
    const userInput = tf.layers.input({
      shape: [1], 
      name: 'User'
    })
    
    const userEmbedding = tf.layers.embedding({
      inputDim: nUsers, 
      outputDim: nFactors,
      name: 'UserEmbedding'
    }).apply(userInput)
    
    const userVec = tf.layers.flatten({
      name: "UserFlat"
    }).apply(userEmbedding)

    // Dot Product of Item and User
    const ratingOutput = tf.layers.dot({
      axes: -1,
      name: "dotProduct"
    }).apply([itemVec, userVec])
    
    // Create the model based on the inputs.
    const model = tf.model({
      inputs: [itemInput, userInput], 
      outputs: ratingOutput,
      name: "Baseline"
    });
  
    console.log("createmodel run")
    return model;
  }


  /**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data    
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const items = data.map(d => d.items)
      const users = data.map(d => d.users)
      const ratings = data.map(d => d.ratings);
  
      const itemsTensor = tf.tensor2d(items, [items.length, 1]);
      const usersTensor = tf.tensor2d(users, [users.length, 1]);
      const ratingsTensor = tf.tensor2d(ratings, [ratings.length, 1]);
  
      //Step 3. Mean center the data
      const ratingsMean = ratingsTensor.mean();
      //const centeredRatings = ratingsTensor.sub(ratingsMean)
      
      return {
        items: itemsTensor,
        users: usersTensor,
        ratings: ratingsTensor,
        // Return the mean bounds so we can use them later.
        ratingsMean: ratingsMean
      }
    });  
  }


  async function trainModel(model, items, users, ratings) {
    // Prepare the model for training.  

    const learningRate = 0.01

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 10;
    const epochs = 2;
    
    console.log("training-start")
    return await model.fit([items, users], ratings, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });

  }

   function getEmbeddings(model){
      return model.getWeights()
  }


  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
    
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
      
      const xs = tf.linspace(0, 1, 100);      
      const preds = model.predict(xs.reshape([100, 1]));      
      
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
      
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
    
   
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
    
    const originalPoints = inputData.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));
    
    
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'}, 
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
  }

  async function run() {

    (async() => {
      await tf.ready 
     // then do all operations on the backend
     console.log("tf ready")
     console.log(tf.getBackend())
    })()
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    console.log(data)
    renderVis(data, "#view")

    const nUsers = 20
    const nItems = 13
    const nFactors = 5

    // Create the model
    const model = createModel(nItems, nUsers, nFactors);  
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    
    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    console.log(tensorData)
    const {items, users, ratings, ratingsMean} = tensorData;
    console.log(items)
        
    // Train the model  
    await trainModel(model, items, users, ratings, ratingsMean);
    //weights = model.getWeights();


    const weights =  getEmbeddings(model)
    console.log(weights[0].dataSync())

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
    // More code will be added below
  }
  
  document.addEventListener('DOMContentLoaded', run);