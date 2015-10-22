var nlp = require("nlp_compromise");
var R = require('./src/recurrent.js');
var Rvis = require('./src/vis.js');
var fs = require('fs');
var natural = require('natural');
var moment = require('moment');

// Model parameters 
var sample_softmax_temperature = Math.pow(10, 0.5); // how peaky model predictions should be
// var sample_softmax_temperature = 1.0;
var generator = 'lstm'; // can also be rnn
var max_chars_gen = 100; // max length of generated sentences
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letter_size = 5;
var hidden_layers = 3;
// TODO: change this variables to one number
var hidden_sizes = [64,64,64]; // list of sizes of hidden layers
var regc = 0.000001; // L2 regularization strength
var learning_rate = 0.01; // learning rate
var clipval = 5.0;
// output params
var total = ''; 
var totalSample = '';

//file sources
var trainingPath = '/shakespeare.txt';
var modelPath = '/output' + trainingPath.slice(0, trainingPath.length-4) + 'model.txt';
var samplePath = '/output' + trainingPath.slice(0, trainingPath.length-4) + 'samples.txt';
var argmaxPath = '/output' + trainingPath.slice(0, trainingPath.length-4) + 'argmax.txt';

// Global variables
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var trainingSet = '';
var solver = new R.Solver(); // should be class because it needs memory for step caches
var pplGraph = new Rvis.Graph();

var model = {};
var tokenizer = new natural.RegexpTokenizer({pattern: /( |\w+|\!|\'|\"|\n)/i});

var initVocab = function(sents, count_threshold) {
  // go over all words and keep track of all unique ones seen
  // join all the sentences
  // var fullText = sents.join(''); 
  var tokens = tokenizer.tokenize(sents);
  // count up all words
  var wordCount = {};
  // special chars, also considered as words 
  for(var i=0,n=tokens.length;i<n;i++) {
    var txti = tokens[i];
    if(txti in wordCount) { wordCount[txti] += 1; } 
    else { wordCount[txti] = 1; }
  }
  // filter by count threshold and create pointers
  // NOTE: start at one because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  var q = 1; 
  for(word in wordCount) {
    if(wordCount[word] >= count_threshold) {
      // add character to vocab
      letterToIndex[word] = q;
      indexToLetter[q] = word;
      vocab.push(word);
      q++;
    }
  }
  input_size = vocab.length + 1;
  output_size = vocab.length + 1;
  epoch_size = sents.length;
}

var utilAddToModel = function(modelto, modelfrom) {
  for(var k in modelfrom) {
    if(modelfrom.hasOwnProperty(k)) {
      // copy over the pointer but change the key to use the append
      modelto[k] = modelfrom[k];
    }
  }
};

var initModel = function() {
  // letter embedding vectors
  var model = {};
  model['Wil'] = new R.RandMat(input_size, letter_size , 0, 0.08);
  
  if(generator === 'rnn') {
    var rnn = R.initRNN(letter_size, hidden_sizes, output_size);
    utilAddToModel(model, rnn);
  } else {
    var lstm = R.initLSTM(letter_size, hidden_sizes, output_size);
    utilAddToModel(model, lstm);
  }

  return model;
};

var reinit = function() {
  // note: reinit writes global vars
  solver = new R.Solver(); // reinit solver
  pplGraph = new Rvis.Graph();

  ppl_list = [];
  tick_iter = 0;
  // read in txt file 
  trainingSet = fs.readFileSync(__dirname + trainingPath, 'utf8');

  // check if there is a jsonfile 
  fs.stat(__dirname + modelPath, function(err, result) {
    if(err) {
      console.log('creating new model');
      initVocab(trainingSet, 1); // takes count threshold for characters
      model = initModel();
      setInterval(tick, 1000);
    } else {
      console.log('loading model');
      model = JSON.parse(fs.readFileSync(__dirname + modelPath));
      loadModel(model);
      setInterval(tick, 1000);
    }
  }); 
}

var saveModel = function() {
  var out = {};
  out['hidden_sizes'] = hidden_sizes;
  out['generator'] = generator;
  out['letter_size'] = letter_size;
  var model_out = {};
  for(var k in model) {
    if(model.hasOwnProperty(k)) {
      model_out[k] = model[k].toJSON();
    }
  }
  out['model'] = model_out;
  var solver_out = {};
  solver_out['decay_rate'] = solver.decay_rate;
  solver_out['smooth_eps'] = solver.smooth_eps;
  step_cache_out = {};
  for(var k in solver.step_cache) {
    if(solver.step_cache.hasOwnProperty(k)) {
      step_cache_out[k] = solver.step_cache[k].toJSON();
    }
  }
  solver_out['step_cache'] = step_cache_out;
  out['solver'] = solver_out;
  out['letterToIndex'] = letterToIndex;
  out['indexToLetter'] = indexToLetter;
  out['vocab'] = vocab;
  fs.writeFileSync(__dirname + modelPath, JSON.stringify(out), 'utf8');
}

var loadModel = function(j) {
  hidden_sizes = j.hidden_sizes;
  generator = j.generator;
  letter_size = j.letter_size;
  model = {};
  for(var k in j.model) {
    if(j.model.hasOwnProperty(k)) {
      var matjson = j.model[k];
      model[k] = new R.Mat(1,1);
      model[k].fromJSON(matjson);
    }
  }
  solver = new R.Solver(); // have to reinit the solver since model changed
  solver.decay_rate = j.solver.decay_rate;
  solver.smooth_eps = j.solver.smooth_eps;
  solver.step_cache = {};
  for(var k in j.solver.step_cache){
      if(j.solver.step_cache.hasOwnProperty(k)){
          var matjson = j.solver.step_cache[k];
          solver.step_cache[k] = new R.Mat(1,1);
          solver.step_cache[k].fromJSON(matjson);
      }
  }
  letterToIndex = j['letterToIndex'];
  indexToLetter = j['indexToLetter'];
  vocab = j['vocab'];

  // reinit these
  ppl_list = [];
  tick_iter = 0;
}

var forwardIndex = function(G, model, ix, prev) {
  var x = G.rowPluck(model['Wil'], ix);
  // forward prop the sequence learner
  if(generator === 'rnn') {
    var out_struct = R.forwardRNN(G, model, hidden_sizes, x, prev);
  } else {
    var out_struct = R.forwardLSTM(G, model, hidden_sizes, x, prev);
  }
  return out_struct;
}

var predictSentence = function(model, samplei, temperature) {
  // console.log('in predict sentence');
  if(typeof samplei === 'undefined') { samplei = false; }
  if(typeof temperature === 'undefined') { temperature = 1.0; }
  var G = new R.Graph(false);
  var s = '';
  var prev = {};
  while(true) {
    var tokens = tokenizer.tokenize(s);
    // RNN tick
    console.log(s);
    console.log(tokens);
    var ix = s.length === 0 ? 0 : letterToIndex[tokens[tokens.length - 1]];
    var lh = forwardIndex(G, model, ix, prev);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
    if(temperature !== 1.0 && samplei) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for(var q=0,nq=logprobs.w.length;q<nq;q++) {
        logprobs.w[q] /= temperature;
      }
    }

    probs = R.softmax(logprobs);
    if(samplei) {
      var ix = R.samplei(probs.w);
    } else {
      var ix = R.maxi(probs.w);  
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(s.length > max_chars_gen) { break; } // something is wrong
    var letter = indexToLetter[ix];
    // console.log('added', letter, 'endadded');
    s += letter + ' ';
  }
  return s;
};

var costfun = function(model, sent) {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  // adjust constfunc for words 
  var tokens = tokenizer.tokenize(sent);
  // console.log(sent);
  var n = tokens.length;
  var G = new R.Graph();
  var log2ppl = 0.0;
  var cost = 0.0;
  var prev = {};
  for(var i=-1; i<n; i++) {
    // start and end tokens are zeros
    var ix_source = i === -1 ? 0 : letterToIndex[tokens[i]]; // first step: start with START token
    var ix_target = i === n-1 ? 0 : letterToIndex[tokens[i+1]]; // last step: end with END token
    lh = forwardIndex(G, model, ix_source, prev);
    prev = lh;

    // set gradients into logprobabilities
    logprobs = lh.o; // interpret output as logprobs
    probs = R.softmax(logprobs); // compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);

    // write gradients into log probabilities
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1;
  }
  var ppl = Math.pow(2, log2ppl / (n - 1));
  return {'G':G, 'ppl':ppl, 'cost':cost};
}

function median(values) {
  values.sort( function(a,b) {return a - b;} );
  var half = Math.floor(values.length/2);
  if(values.length % 2) return values[half];
  else return (values[half-1] + values[half]) / 2.0;
}

var ppl_list = [];
var tick_iter = 0;

var tick = function() {

  // sample sentence fromd data
  // split out trainingSet on /n
  var lines = trainingSet.split('\n');
  var sentix = R.randi(0,lines.length);
  var sent = lines[sentix + 1] === '' ? lines[sentix] + '\n\n' : lines[sentix] + '\n';
  // select randon sentence 
  // add newline at end

  var t0 = +new Date();  // log start timestamp
  // evaluate cost function on a sentence
  var cost_struct = costfun(model, sent);
  
  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  // perform param update
  var solver_stats = solver.step(model, learning_rate, regc, clipval);
  //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)

  var t1 = +new Date();
  var tick_time = t1 - t0;

  ppl_list.push(cost_struct.ppl); // keep track of perplexity

  // evaluate now and then
  tick_iter += 1;
  var temperatures = [0.2, 0.4, 0.6, 1, 1.4, 1.8, 2.5, 2.8, 3.2, 4];
  if(tick_iter % 5 === 0) {
    totalSample += '\n ============'+ moment().format('MMMM Do YYYY, hh:mm:ss a') +'============= \n';
    for (var i = 0; i < temperatures.length; i++) {
      var pred2 = '\n\n' + 'xxxxxxxxxxxxx' + temperatures[i] + 'xxxxxxxxxxxxxxxx' + '\n';
      for(var q=0;q<5;q++) {
        pred2 += predictSentence(model, true, temperatures[i]);
        totalSample += pred2;
      }
      fs.writeFileSync(__dirname + '/output/samples.txt', totalSample, 'utf8');
    }
  }

  if(tick_iter % 10 === 0) {

    // GREEDY argmax prediction
    var pred = predictSentence(model, false);
    total += '\n ============'+ moment().format('MMMM Do YYYY, hh:mm:ss a') +'============= \n' + 
    'argmax PERPLEXITY: ' + cost_struct.ppl.toFixed(2) + '\n' + pred;
    fs.writeFileSync(__dirname + argmaxPath, total, 'utf8');

    if(tick_iter % 100 === 0) {
      var median_ppl = median(ppl_list);
      ppl_list = [];
      console.log('tick function, graph call');
      pplGraph.add(tick_iter, median_ppl);
      // pplGraph.drawSelf(document.getElementById("pplgraph"));
    }
    // save model
  }
  if(tick_iter % 1200) {
    saveModel();
  }
}
// start training your model
reinit();


