use argmin::core::{CostFunction, Error, Executor, observers::ObserverMode};
use argmin_observer_slog::SlogLogger;
use cobyla_argmin::CobylaSolver;

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

fn main() {
    println!(
        "*** Solve paraboloid problem using Cobyla argmin solver implemented on top of fmin_cobyla impl"
    );
    let problem = ParaboloidProblem;
    let solver = CobylaSolver::new(vec![1., 1.]);

    let res = Executor::new(problem, solver)
        .timer(true)
        .configure(|state| state.max_iters(100).iprint(0))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("*** Result argmin solver impl ***");
    println!("Result:\n{}", res);
}
