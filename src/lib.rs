#![doc = include_str!("../README.md")]

mod cobyla;
mod cobyla_solver;
mod cobyla_state;
pub use crate::cobyla_solver::*;
pub use crate::cobyla_state::*;

/// Failed termination status of the optimization process
#[derive(Debug, Clone, Copy)]
pub enum FailStatus {
    Failure,
    InvalidArgs,
    OutOfMemory,
    RoundoffLimited,
    ForcedStop,
    UnexpectedError,
}

/// Successful termination status of the optimization process
#[derive(Debug, Clone, Copy)]
pub enum SuccessStatus {
    Success,
    StopValReached,
    FtolReached,
    XtolReached,
    MaxEvalReached,
    MaxTimeReached,
}

/// Tolerances used as termination criteria.
/// For all, condition is disabled if value is not strictly positive.
/// ```rust
/// # use crate::cobyla_argmin::StopTols;
/// let stop_tol = StopTols {
///     ftol_rel: 1e-4,
///     xtol_abs: vec![1e-3; 3],   // size should be equal to x dim
///     ..StopTols::default()      // default stop conditions are disabled
/// };  
/// ```
#[derive(Debug, Clone, Default)]
pub struct StopTols {
    /// Relative tolerance on function value, algorithm stops when `func(x)` changes by less than `ftol_rel * func(x)`
    pub ftol_rel: f64,
    /// Absolute tolerance on function value, algorithm stops when `func(x)` change is less than `ftol_rel`
    pub ftol_abs: f64,
    /// Relative tolerance on optimization parameters, algorithm stops when all `x[i]` changes by less than `xtol_rel * x[i]`
    pub xtol_rel: f64,
    /// Relative tolerance on optimization parameters, algorithm stops when `x[i]` changes by less than `xtol_abs[i]`
    pub xtol_abs: Vec<f64>,
}

/// An enum for specifying the initial change of x which correspond to the `rhobeg`
/// argument of the original Powell's algorithm (hence the name)
pub enum RhoBeg {
    /// Used when all x components changes are specified with a single given value
    All(f64),
    /// Used to set the components with the given x-dim-sized vector
    Set(Vec<f64>),
}

#[cfg(test)]
mod tests {
    use crate::CobylaSolver;
    use approx::assert_abs_diff_eq;
    use argmin::core::{CostFunction, Error, Executor, State};

    /// Problem cost function
    fn paraboloid(x: &[f64], _data: &mut ()) -> f64 {
        10. * (x[0] + 1.).powf(2.) + x[1].powf(2.)
    }

    /// Problem Definition for CobylaSolver : minimize paraboloid(x) subject to x0 >= 0
    struct ParaboloidProblem;

    impl CostFunction for ParaboloidProblem {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            let fx = paraboloid(x, &mut ());
            Ok(vec![fx, x[0]])
        }
    }

    #[test]
    fn test_paraboloid() {
        let problem = ParaboloidProblem;
        let solver = CobylaSolver::new(vec![1., 1.]);

        let res = Executor::new(problem, solver)
            .timer(true)
            .configure(|state| state.max_iters(100).iprint(0))
            .run()
            .unwrap();

        assert_abs_diff_eq!(0., res.state().get_best_param().unwrap()[0], epsilon = 1e-2);
        assert_abs_diff_eq!(0., res.state().get_best_param().unwrap()[1], epsilon = 1e-2);
        assert_abs_diff_eq!(10., res.state().get_best_cost(), epsilon = 1e-2);
    }
}
