import checkOptions from './checkOptions';
import step from './step';

/**
 * Curve fitting algorithm
 * @param {{x:ArrayLike<number>, y:ArrayLike<number>}} data - Array of points to fit in the format [x1, x2, ... ], [y1, y2, ... ]
 * @param {function} parameterizedFunction - Takes an array of parameters and returns a function with the independent variable as its sole argument
 * @param {object} options - Options object
 * @param {ArrayLike<number>} options.initialValues - Array of initial parameter values
 * @param {number|ArrayLike<number>} [options.weights = 1] - weighting vector, if the length does not match with the number of data points, the vector is reconstructed with first value.
 * @param {number} [options.damping = 1e-2] - Levenberg-Marquardt parameter, small values of the damping parameter λ result in a Gauss-Newton update and large
values of λ result in a gradient descent update
 * @param {number} [options.dampingStepDown = 9] - factor to reduce the damping (Levenberg-Marquardt parameter) when there is not an improvement when updating parameters.
 * @param {number} [options.dampingStepUp = 11] - factor to increase the damping (Levenberg-Marquardt parameter) when there is an improvement when updating parameters.
 * @param {number} [options.improvementThreshold = 1e-3] - the threshold to define an improvement through an update of parameters
 * @param {number|ArrayLike<number>} [options.gradientDifference = 10e-2] - The step size to approximate the jacobian matrix
 * @param {boolean} [options.centralDifference = false] - If true the jacobian matrix is approximated by central differences otherwise by forward differences
 * @param {ArrayLike<number>} [options.minValues] - Minimum allowed values for parameters
 * @param {ArrayLike<number>} [options.maxValues] - Maximum allowed values for parameters
 * @param {number} [options.maxIterations = 100] - Maximum of allowed iterations
 * @param {number} [options.errorTolerance = 10e-3] - Minimum uncertainty allowed for each point.
 * @return {{parameterValues: Array<number>, parameterError: number, iterations: number}}
 */
export class levenbergMarquardt {
  constructor(data, parameterizedFunction, options) {
    let opts = checkOptions(data, parameterizedFunction, options);
    this.data = data;
    this.parameterizedFunction = parameterizedFunction;

    this.minValues = opts.minValues;
    this.maxValues = opts.maxValues;
    this.parameters = opts.parameters;
    this.weightSquare = opts.weightSquare;
    this.damping = opts.damping;
    this.dampingStepUp = opts.dampingStepUp;
    this.dampingStepDown = opts.dampingStepDown;
    this.maxIterations = opts.maxIterations;
    this.errorTolerance = opts.errorTolerance;
    this.errorCalculation = opts.errorCalculation;
    this.centralDifference = opts.centralDifference;
    this.gradientDifference = opts.gradientDifference;
    this.improvementThreshold = opts.improvementThreshold;

    this.iteration = 0;
    this._frame = undefined;
  }

  start() {
    if (!this._fitting) {
      this._fitting = true;

      this.iteration = 0;
      this.error = this.errorCalculation(
        this.data,
        this.parameters,
        this.parameterizedFunction,
        this.weightSquare,
      );
      this.optimalError = this.error;
      this.optimalParameters = this.parameters.slice();
      this.fit();
    }
  }

  stop() {
    this._fitting = false;
    if (this._frame) window.cancelAnimationFrame(this._frame);
  }

  getResults() {
    return {
      parameterValues: this.optimalParameters,
      parameterError: this.optimalError,
      iterations: this.iteration,
    };
  }

  fit() {
    if (this._fitting) {
      this._frame = window.requestAnimationFrame(this.fit);

      let converged = this.error <= this.errorTolerance;

      if (!converged && this.iteration < this.maxIterations) {
        this.iteration += 1;
        let previousError = this.error;

        let { perturbations, jacobianWeightResidualError } = step(
          this.data,
          this.parameters,
          this.damping,
          this.gradientDifference,
          this.parameterizedFunction,
          this.centralDifference,
          this.weightSquare,
        );

        for (let k = 0; k < this.parameters.length; k++) {
          this.parameters[k] = Math.min(
            Math.max(
              this.minValues[k],
              this.parameters[k] - perturbations.get(k, 0),
            ),
            this.maxValues[k],
          );
        }

        this.error = this.errorCalculation(
          this.data,
          this.parameters,
          this.parameterizedFunction,
          this.weightSquare,
        );

        if (isNaN(this.error)) {
          throw new Error('Error should be real value: ' + this.error);
        }

        if (this.error < this.optimalError - this.errorTolerance) {
          this.optimalError = this.error;
          this.optimalParameters = this.parameters.slice();
        }

        let improvementMetric =
          (previousError - this.error) /
          perturbations
            .transpose()
            .mmul(
              perturbations.mul(this.damping).add(jacobianWeightResidualError),
            )
            .get(0, 0);

        if (improvementMetric > this.improvementThreshold) {
          this.damping = Math.max(this.damping / this.dampingStepDown, 1e-7);
        } else {
          this.damping = Math.min(this.damping * this.dampingStepUp, 1e7);
        }
      }
    }
  }
}
