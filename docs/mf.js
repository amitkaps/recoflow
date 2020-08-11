import {getData} from "/recoflowjs/data.js"
import {renderVis, renderEmbedding, strip} from "/recoflowjs/vis.js"
import {getLayers, getEmbeddings} from "/recoflowjs/inspect.js"
import {modelMF, trainModel} from "/recoflowjs/model.js"
import {makeTensors} from "/recoflowjs/preprocess.js"

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
    const nFactors = 2

    // Create the model
    const model = modelMF(nItems, nUsers, nFactors);  
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    
    // Convert the data to a form we can use for training.
    const tensorData = makeTensors(data);
    console.log(tensorData)
    const {items, users, ratings, ratingsMean} = tensorData;
    console.log(items)
        
    // Train the model  
    await trainModel(model, items, users, ratings, ratingsMean);
    //weights = model.getWeights();


    const layers = getLayers(model)
    console.log(layers)
    const embeddingTensors =  await model.getWeights()
    
    const e0 = embeddingTensors[0]
    const e1 = embeddingTensors[1]
    
    console.log(e1.transpose().shape)
    const reconstruction = e1.dot(e0.transpose())
    console.log(reconstruction.shape)
    
    const embedding0 = embeddingTensors[0].arraySync()
    const embedding1 = embeddingTensors[1].arraySync()
    console.log(embedding0)
    console.log(embedding1)


    renderEmbedding(embedding0, "#embedding0")
    renderEmbedding(embedding1, "#embedding1")

    console.log(tf.memory())
    // Make some predictions using the model and compare them to the
    // original data
    // testModel(model, data, tensorData);
    // More code will be added below
  }
  
  document.addEventListener('DOMContentLoaded', run);