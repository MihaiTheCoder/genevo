use crate::{
    algorithm::{Algorithm, MemReuse},
    random::{get_rng, random_seed, Prng, Seed},
    simulation::{SimResult, Simulation, SimulationBuilder, State},
    termination::{StopFlag, Termination}, genetic::Genotype,
};
use chrono::{DateTime, Local};
use std::{
    error::Error,
    fmt::{self, Debug, Display},
    hash::Hash, marker::PhantomData,
};

/// The `simulate` function creates a new `Simulator` for the given
/// `algorithm::Algorithm`.
pub fn simulate<A, G>(algorithm: A) -> SimulatorBuilderWithAlgorithm<A,G>
where
    A: Algorithm<G>,
    G: Genotype
{
    SimulatorBuilderWithAlgorithm { algorithm, _phantom: PhantomData }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulatorBuilder<A, T, G>
where
    A: Algorithm<G>,
    T: Termination<A, G>,
    G:Genotype
{
    algorithm: A,
    termination: T,
    _phantom: PhantomData<G>
}

impl<A, T, G> SimulationBuilder<Simulator<A, T, G>, A, G> for SimulatorBuilder<A, T, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: Eq + Hash + Display + Send + Sync,
    T: Termination<A, G>,
    G: Genotype
{
    fn build(self) -> Simulator<A, T, G> {
        self.build_with_seed(random_seed())
    }

    fn build_with_seed(self, seed: Seed) -> Simulator<A, T, G> {
        Simulator {
            algorithm: self.algorithm,
            termination: self.termination,
            run_mode: RunMode::NotRunning,
            rng: get_rng(seed),
            started_at: Local::now(),
            iteration: 0,
            mem_reuse: MemReuse::new()
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulatorBuilderWithAlgorithm<A, G>
where
    A: Algorithm<G>,
    G: Genotype
{
    algorithm: A,
    _phantom: PhantomData<G>
}

impl<A, G> SimulatorBuilderWithAlgorithm<A, G>
where
    A: Algorithm<G>,
    G: Genotype
{
    pub fn until<T>(self, termination: T) -> SimulatorBuilder<A, T, G>
    where
        T: Termination<A, G>,
    {
        SimulatorBuilder {
            algorithm: self.algorithm,
            termination,
            _phantom:PhantomData
        }
    }
}

/// The `RunMode` identifies whether the simulation is running and how it has
/// been started.
#[derive(Clone, Debug, PartialEq)]
enum RunMode {
    /// The simulation is running in loop mode. i.e. it was started by calling
    /// the `run` function.
    Loop,
    /// The simulation is running in step mode. i.e. it was started by calling
    /// the `step` function.
    Step,
    /// The simulation is not running.
    NotRunning,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SimError<A, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: Eq + Hash + Debug,
    G: Genotype
{
    AlgorithmError(<A as Algorithm<G>>::Error),
    SimulationAlreadyRunning(String),
}

impl<A, G> Display for SimError<A, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: Eq + Hash + Debug + Display,
    G: Genotype
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SimError::AlgorithmError(ref error) => write!(f, "algorithm error: {}", error),
            SimError::SimulationAlreadyRunning(ref message) => {
                write!(f, "simulation already running {}", message)
            }
        }
    }
}

impl<A, G> Error for SimError<A, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: 'static + Eq + Hash + Debug + Display,
    G: Genotype
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            SimError::AlgorithmError(ref error) => Some(error),
            SimError::SimulationAlreadyRunning(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Simulator<A, T, G>
where
    G: Genotype,
    A: Algorithm<G>,
    T: Termination<A, G>,
    
{
    algorithm: A,
    termination: T,
    run_mode: RunMode,
    rng: Prng,
    started_at: DateTime<Local>,
    iteration: u64,
    mem_reuse: MemReuse<G>
}

impl<A, T, G> Simulator<A, T, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: Eq + Hash + Display + Send + Sync,
    T: Termination<A, G>,
    G: Genotype
{

    pub fn termination(&self) -> &T {
        &self.termination
    }

    /// Processes one iteration of the algorithm used in this simulation.
    fn process_one_iteration(&mut self) -> Result<State<A, G>, <Self as Simulation<A, G>>::Error> {
        let loop_started_at = Local::now();

        self.iteration += 1;
        let result = self.algorithm.next(self.iteration, &mut self.rng, &mut self.mem_reuse);

        let loop_duration = Local::now().signed_duration_since(loop_started_at);
        match result {
            Ok(result) => Ok(State {
                started_at: self.started_at,
                iteration: self.iteration,
                duration: loop_duration,
                result,
            }),
            Err(error) => Err(SimError::AlgorithmError(error)),
        }
    }
}

impl<A, T, G> Simulation<A, G> for Simulator<A, T, G>
where
    A: Algorithm<G> + Debug,
    <A as Algorithm<G>>::Error: Eq + Hash + Display + Send + Sync,
    T: Termination<A, G>,
    G: Genotype
{
    type Error = SimError<A, G>;

    fn run(&mut self) -> Result<SimResult<A, G>, Self::Error> {
        match self.run_mode {
            RunMode::Loop => {
                return Err(SimError::SimulationAlreadyRunning(format!(
                    "in loop mode since {}",
                    &self.started_at
                )))
            }
            RunMode::Step => {
                return Err(SimError::SimulationAlreadyRunning(format!(
                    "in step mode since {}",
                    &self.started_at
                )))
            }
            RunMode::NotRunning => {
                self.run_mode = RunMode::Loop;
                self.started_at = Local::now();
            }
        }
        let result = loop {
            match self.process_one_iteration() {
                Ok(state) => {
                    // Stage 5: Be aware of the termination:
                    match self.termination.evaluate(&state) {
                        StopFlag::Continue => {}
                        StopFlag::StopNow(reason) => {
                            let duration = Local::now().signed_duration_since(self.started_at);
                            break Ok(SimResult::Final(state, duration, reason));
                        }
                    }
                }
                Err(error) => {
                    break Err(error);
                }
            }
        };
        self.run_mode = RunMode::NotRunning;
        result
    }

    fn step(&mut self) -> Result<SimResult<A, G>, Self::Error> {
        match self.run_mode {
            RunMode::Loop => {
                return Err(SimError::SimulationAlreadyRunning(format!(
                    "in loop mode since {}",
                    &self.started_at
                )))
            }
            RunMode::Step => (),
            RunMode::NotRunning => {
                self.run_mode = RunMode::Step;
                self.started_at = Local::now();
            }
        }
        self.process_one_iteration().and_then(|state|
            // Stage 5: Be aware of the termination:
            Ok(match self.termination.evaluate(&state) {
                StopFlag::Continue => {
                    SimResult::Intermediate(state)
                },
                StopFlag::StopNow(reason) => {
                    let duration = Local::now().signed_duration_since(self.started_at);
                    self.run_mode = RunMode::NotRunning;
                    SimResult::Final(state, duration, reason)
                },
            }))
    }

    fn stop(&mut self) -> Result<bool, Self::Error> {
        match self.run_mode {
            RunMode::Loop | RunMode::Step => {
                self.run_mode = RunMode::NotRunning;
                Ok(true)
            }
            RunMode::NotRunning => Ok(false),
        }
    }

    fn reset(&mut self) -> Result<bool, Self::Error> {
        match self.run_mode {
            RunMode::Loop => {
                return Err(SimError::SimulationAlreadyRunning(format!(
                    "Simulation still running in loop mode since {}. Wait for the \
                     simulation to finish or stop it before resetting it.",
                    &self.started_at
                )))
            }
            RunMode::Step => {
                return Err(SimError::SimulationAlreadyRunning(format!(
                    "Simulation still running in step mode since {}. Wait for the \
                     simulation to finish or stop it before resetting it.",
                    &self.started_at
                )))
            }
            RunMode::NotRunning => (),
        }
        self.run_mode = RunMode::NotRunning;
        self.iteration = 0;
        self.algorithm.reset().map_err(SimError::AlgorithmError)
    }
}
