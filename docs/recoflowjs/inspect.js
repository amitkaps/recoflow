function getEmbeddings(model){
    return model.getWeights()
}

function getLayers(model){
    return model.layers
}


export {getEmbeddings, getLayers}