class ActivationFunction {
  /**
   * @param {Function} f
   * @param {Function} df
   */
  constructor(f, df) {
    this.f = f;
    this.df = df;
  }
}

const sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

const tanh = new ActivationFunction(x => Math.tanh(x), y => 1 - y * y);

class n_Error {
  /**
   *
   * @param {Matrix} target
   * @param {Matrix} prediction
   */
  static meanSquaredError(target, prediction) {
    return target.subtract(prediction);
  }
}

class Layer {
  /**
   * Create a new Layer
   * @param {number} input
   * @param {ActivationFunction} activationFunction
   */
  constructor(nodes, activationFunction) {
    this.nodes = nodes;
    this.activationFunction = activationFunction;
    this.data = new Matrix(nodes, 1);
    this.weights;
    this.bias;
  }

  propagate(input) {
    // TODO: Check input size
    this.data = input;
    return Matrix.dot(this.weights, input);
  }
}

class Matrix {
  /**
   * @param { Number } rows
   * @param { Number } cols
   */
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.values = Array(this.rows)
      .fill()
      .map(() => Array(this.cols).fill(0));
    return this;
  }

  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((e, i) => arr[i]);
  }

  /**
   * @param { Matrix } _matrix
   */
  add(_matrix) {
    if (this.rows != _matrix.rows || this.cols != _matrix.cols) {
      throw Error("Rows and cols of matrix must match.");
    }
    this.map((val, i, j) => val + _matrix.values[i][j]);
    return this;
  }

  randomize() {
    return this.map(e => Math.random() * 2 - 1);
  }

  subtract(_matrix) {
    if (this.rows != _matrix.rows || this.cols != _matrix.cols) {
      throw Error("Rows and cols of matrix must match.");
    }
    this.map((val, i, j) => val - _matrix.values[i][j]);
    return this;
  }

  /**
   * @param { Matrix } matrix1
   * @param { Matrix } matrix2
   */
  static subtract(matrix1, matrix2) {
    if (matrix1.rows !== matrix2.rows || matrix1.cols !== matrix2.cols) {
      throw Error("Rows and cols of matrix must match.");
    }
    let result = new Matrix(matrix1.rows, matrix1.cols);
    result.map((val, i, j) => matrix1.values[i][j] - matrix2.values[i][j]);
    return result;
  }

  /**
   * @param { Number | Matrix} n
   */
  scalar(n) {
    if (n instanceof Matrix) {
      this.map((val, i, j) => val * n.values[i][j]);
    } else {
      this.map(val => val * n);
    }
    return this;
  }

  /**
   * @param { Matrix } matrix1
   * @param { Matrix } matrix2
   */
  static dot(matrix1, matrix2) {
    if (matrix1.cols != matrix2.rows) {
      throw Error("Cols of the matrix1 must me equal to rows of matrix2");
    }
    let _m = new Matrix(matrix1.rows, matrix2.cols);
    for (let i = 0; i < _m.rows; i++) {
      for (let j = 0; j < _m.cols; j++) {
        let sum = 0;
        for (let r = 0; r < matrix1.cols; r++) {
          sum += matrix1.values[i][r] * matrix2.values[r][j];
        }
        _m.values[i][j] = sum;
      }
    }
    return _m;
  }

  /**
   * @param { Matrix } matrix
   */
  static transpose(matrix) {
    let _m = new Matrix(matrix.cols, matrix.rows);
    _m.map((val, i, j) => matrix.values[j][i]);
    return _m;
  }

  copy() {
    let m = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        m.values[i][j] = this.values[i][j];
      }
    }
    return m;
  }

  map(f) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.values[i][j] = f(this.values[i][j], i, j);
      }
    }
    return this;
  }

  /**
   *
   * @param {Matrix} matrix
   * @param {Function} f
   */
  static map(matrix, f) {
    return new Matrix(matrix.rows, matrix.cols).map((e, i, j) =>
      f(matrix.values[i][j], i, j)
    );
  }

  toArray() {
    let arr = [];
    this.map(x => arr.push(x));
    return arr;
  }

  print() {
    console.log(this.values);
  }
}

class Model {
  /**
   * Create a new neural network model
   * @param {Function} err
   * @param {Number} learning_rate
   */
  constructor(err, learning_rate) {
    this.err = err;
    this.learning_rate = learning_rate;
    this.layers = [];
  }

  /**
   * Add new layer to the model
   * @param {Layer} layer
   */
  addLayer(layer) {
    if (this.layers.length >= 1) {
      let prev = this.layers[this.layers.length - 1];
      // Intialize the weights
      prev.weights = new Matrix(layer.nodes, prev.nodes).randomize();
      layer.bias = new Matrix(layer.nodes, 1).randomize();
    }
    this.layers.push(layer);
  }

  /**
   * Calculate the output of the network
   * @param {Array} inputs
   */
  predict(input) {
    let output = Matrix.fromArray(input);
    for (let i = 0, len = this.layers.length - 1; i < len; i++) {
      let layer = this.layers[i];
      let next_layer = this.layers[i + 1];
      output = layer.propagate(output);
      output.add(next_layer.bias);
      output.map(next_layer.activationFunction.f);
    }
    this.layers[this.layers.length - 1].data = output;
    return output;
  }

  /**
   * Train the network
   * @param {Array} inputs Array of inputs
   * @param {Array} outputs Array of outputs
   */
  train(inputs, outputs) {
    if (inputs.length !== outputs.length) {
      return Error("Length of inputs must be equal to length of output.");
    }

    let total_error = new Matrix(this.layers[this.layers.length - 1].nodes, 1);

    for (let i = 0; i < inputs.length; i++) {
      let x = inputs[i];
      let y = Matrix.fromArray(outputs[i]);

      let prediction = this.predict(x).copy();

      let final_error = this.err(y, prediction);
      total_error.add(final_error);

      for (let j = this.layers.length - 2; j >= 0; j--) {
        let layer = this.layers[j];
        let next_layer = this.layers[j + 1];

        let hidden_error = final_error.copy();

        let gradient = Matrix.map(
          next_layer.data,
          next_layer.activationFunction.df
        );
        gradient.scalar(hidden_error);
        gradient.scalar(this.learning_rate);

        let deltas = Matrix.dot(gradient, Matrix.transpose(layer.data));
        layer.weights.add(deltas);
        next_layer.bias.add(gradient);

        final_error = Matrix.dot(
          Matrix.transpose(this.layers[j].weights),
          final_error
        );
      }
    }
    let e = 0;
    total_error.map(x => {
      e += Math.abs(x) / total_error.rows;
      return x;
    });
    return e;
  }
}
