

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 */
function makeTensors(data) {
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

  export {makeTensors};