//! This module provides an `algorithm::Algorithm` which implements the genetic
//! algorithm (GA).
//!
//! The stages of the basic genetic algorithm are:
//!
//! 1. **Initialize**: Generate random population of n genotypes (or chromosomes)
//! 2. **Fitness**: Evaluate the fitness of each genotype in the population
//! 3. **New Population**: Create a new population by repeating following steps
//!    until the new population is complete:
//! 3.1. **Selection**: Select a tuple of parent genotypes from a population
//!      according to their fitness and the selection strategy of the
//!      configured `operator::SelectionOp`
//! 3.2. **Crossover**: With a crossover probability cross over the parents to
//!      form a new offspring (child) by means of the configured
//!      `operator::CrossoverOp`.
//! 3.3. **Mutation**: With a mutation probability mutate new offspring at each
//!      locus (position in genotype) by means of the configured
//!      `operator::MutationOp`.
//! 3.4. **Accepting**: Place new offspring in the new population.
//! 4. **Replace**: Use new generated population for a further run of the
//!    algorithm.
//! 5. **Termination**: If the end condition is satisfied, stop, and return the
//!    best solution in current population.
//! 6. **Loop**: Go to step 2

pub mod builder;

use self::builder::EmptyGeneticAlgorithmBuilder;
use crate::{
    algorithm::{Algorithm, BestSolution, EvaluatedPopulation, Evaluated, MemReuse},
    genetic::{Fitness, FitnessFunction, Genotype, Offspring, Parents},
    operator::{CrossoverOp, MutationOp, ReinsertionOp, SelectionOp},
    population::Population,
    random::Prng,
};
use chrono::Local;
use std::{
    fmt::{self, Display},
    marker::PhantomData,
};

/// The `State` struct holds the results of one pass of the genetic algorithm
/// loop, i.e. the processing of the evolution from one generation to the next
/// generation.
#[derive(Clone, Debug, PartialEq)]
pub struct State<G, F>
where
    G: Genotype,
    F: Fitness,
{
    /// The evaluated population of the current generation.
    pub evaluated_population: EvaluatedPopulation<G, F>,
    /// Best solution of this generation.
    pub best_solution: BestSolution<G, F>,
}

/// An error that can occur during execution of a `GeneticAlgorithm`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GeneticAlgorithmError {
    /// The algorithm is run with an empty population.
    EmptyPopulation(String),
    /// The algorithm is run with an population size that is smaller than the
    /// required minimum.
    PopulationTooSmall(String),
}

impl Display for GeneticAlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GeneticAlgorithmError::EmptyPopulation(details) => write!(f, "{}", details),
            GeneticAlgorithmError::PopulationTooSmall(details) => write!(f, "{}", details),
        }
    }
}

impl std::error::Error for GeneticAlgorithmError {}

pub fn genetic_algorithm<G, F>() -> EmptyGeneticAlgorithmBuilder<G, F>
where
    G: Genotype,
    F: Fitness,
{
    EmptyGeneticAlgorithmBuilder::new()
}

/// A `GeneticAlgorithm` declares the building blocks that make up the actual
/// algorithm for a specific optimization problem.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneticAlgorithm<G, F, E, S, C, M, R>
where
    G: Genotype,
    F: Fitness,
    E: FitnessFunction<G, F>,
    S: SelectionOp<G, F>,
    C: CrossoverOp<G>,
    M: MutationOp<G>,
    R: ReinsertionOp<G, F>,
{
    _f: PhantomData<F>,
    evaluator: E,
    selector: S,
    breeder: C,
    mutator: M,
    reinserter: R,
    min_population_size: usize,
    initial_population: Population<G>,
    population: Vec<G>
}

impl<G, F, E, S, C, M, R> GeneticAlgorithm<G, F, E, S, C, M, R>
where
    G: Genotype,
    F: Fitness,
    E: FitnessFunction<G, F>,
    S: SelectionOp<G, F>,
    C: CrossoverOp<G>,
    M: MutationOp<G>,
    R: ReinsertionOp<G, F>,
{
    pub fn evaluator(&self) -> &E {
        &self.evaluator
    }

    pub fn selector(&self) -> &S {
        &self.selector
    }

    pub fn breeder(&self) -> &C {
        &self.breeder
    }

    pub fn mutator(&self) -> &M {
        &self.mutator
    }

    pub fn reinserter(&self) -> &R {
        &self.reinserter
    }

    pub fn min_population_size(&self) -> usize {
        self.min_population_size
    }
}

impl<G, F, E, S, C, M, R> Algorithm<G> for GeneticAlgorithm<G, F, E, S, C, M, R>
where
    G: Genotype,
    F: Fitness + Send + Sync,
    E: FitnessFunction<G, F> + Sync,
    S: SelectionOp<G, F>,
    C: CrossoverOp<G> + Sync,
    M: MutationOp<G> + Sync,
    R: ReinsertionOp<G, F>,
{
    type Output = State<G, F>;
    type Error = GeneticAlgorithmError;

    fn next(&mut self, iteration: u64, rng: &mut Prng, mem_reuse: &mut MemReuse<G> ) -> Result<Self::Output, Self::Error> {
        if self.population.is_empty() {
            return Err(GeneticAlgorithmError::EmptyPopulation(format!(
                "Population of generation {} is empty. The required minimum size for \
                 populations is {}.",
                iteration, self.min_population_size
            )));
        }
        if self.population.len() < self.min_population_size {
            return Err(GeneticAlgorithmError::PopulationTooSmall(format!(
                "Population of generation {} has a size of {} which is smaller than the \
                 required minimum size of {}",
                iteration,
                self.population.len(),
                self.min_population_size
            )));
        }

        // Stage 2: The fitness check:
        let evaluation = evaluate_fitness(self.population.clone(), &self.evaluator);
        
        let best_solution = BestSolution {
            found_at: Local::now(),
            generation: iteration,
            solution: evaluation.best_evaluation.clone(),
        };

        // Stage 3: The making of a new population:
        let counts = self.selector.get_counts(evaluation.individuals().len());
        let (mut breeding, mut selection) =  mem_reuse.get_breeding_and_parents(counts.num_of_parents, counts.num_individuals_per_parent);
        self.selector.select_from(&evaluation, rng, counts, &mut selection);

        
        par_breed_offspring(selection, &self.breeder, &self.mutator, rng, &mut breeding);
        let reinsertion = self.reinserter.combine(&mut breeding, &evaluation, rng);
        mem_reuse.clear_breeding_and_selections();

        // Stage 4: On to the next generation:
        let next_generation = reinsertion;
        self.population = next_generation;
        Ok(State {
            evaluated_population: evaluation,
            best_solution: best_solution,
        })
    }

    fn reset(&mut self) -> Result<bool, Self::Error> {
        self.population = self.initial_population.individuals().to_vec();
        Ok(true)
    }
}

fn evaluate_fitness<G, F, E>(
    population: Vec<G>,
    evaluator: &E,
) -> EvaluatedPopulation<G, F>
where
    G: Genotype + Sync,
    F: Fitness + Send + Sync,
    E: FitnessFunction<G, F> + Sync,
{
    let evaluation = par_evaluate_fitness(&population, evaluator);
    let average = evaluator.average(&evaluation.0);
    let evaluated = EvaluatedPopulation::new(
        population,
        evaluation.0,
        evaluation.1,
        evaluation.2,        
        average,
        evaluation.3
    );
    evaluated
}

fn par_evaluate_fitness<G, F, E>(population: &[G], evaluator: &E) -> (Vec<F>, F, F, Evaluated<G, F>)
where
    G: Genotype + Sync,
    F: Fitness + Send + Sync,
    E: FitnessFunction<G, F> + Sync,
{    
    let mut fitness = Vec::with_capacity(population.len());
    let mut highest = evaluator.lowest_possible_fitness();
    let mut lowest = evaluator.highest_possible_fitness();
    let mut highest_genome = &population[0];
    for genome in population.iter() {
        let score = evaluator.fitness_of(genome);
        if score > highest {
            highest = score.clone();
            highest_genome = genome;
        }
        if score < lowest {
            lowest = score.clone();
        }
        fitness.push(score);
    }
    let evaluated = Evaluated {fitness: highest.clone(), genome: highest_genome.clone()};
    

    (fitness, highest, lowest, evaluated)

}

/// Lets the parents breed their offspring and mutate its children. And
/// finally combines the offspring of all parents into one big offspring.

fn par_breed_offspring<G, C, M>(
    parents: &mut Vec<Parents<G>>,
    breeder: &C,
    mutator: &M,
    rng: &mut Prng,
    offspring: &mut Offspring<G>
)
where
    G: Genotype + Send,
    C: CrossoverOp<G> + Sync,
    M: MutationOp<G> + Sync,
{
    for parents in parents {
        let children = breeder.crossover(parents, rng);
        for child in children {
            let mutated = mutator.mutate(child, rng);
            offspring.push(mutated);
        }
    }
}