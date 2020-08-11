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



  function reduceArrayPrecision (array) {
    
  }


  export {getData};
