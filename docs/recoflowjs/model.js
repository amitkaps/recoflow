/**
*  
**/


function modelMF(nItems, nUsers, nFactors) {

    // Item Layer
    const itemInput = tf.layers.input({
        shape: [1], 
        name: 'Item'
    })

    const itemEmbedding = tf.layers.embedding({
        inputDim: nItems, 
        outputDim: nFactors,
        name: 'ItemEmbedding',
        inputLength: 1
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

async function trainModel(model, items, users, ratings) {
    // Prepare the model for training.  

    const learningRate = 0.01

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 10;
    const epochs = 20;
    
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


export {modelMF, trainModel}