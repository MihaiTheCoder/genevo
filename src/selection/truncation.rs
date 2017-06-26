//! The `truncation` module provides selection methods that are merely based
//! on the fitness values of the individuals but don't have any stochastic
//! component.
//!
//! The provided `SelectionOp` implementations are:
//! * `MaximizeSelector`

use genetic::{Fitness, Genotype, Parents};
use operator::{GeneticOperator, SelectionOp, SingleObjective, MultiObjective};
use simulation::{EvaluatedPopulation, SimError};


/// The `MaximizeSelector` selects the best performing `genetic::Genotype`s
/// from the population.
///
/// This `MaximizeSelector` can be used for single-objective fitness values
/// as well as multi-objective fitness values.
#[derive(Clone)]
pub struct MaximizeSelector {
    /// The truncation threshold is the ratio between the number of parents
    /// to be selected and the size of the population:
    /// threshold = number of parents / size of population
    truncation_threshold: f64,
    /// The number of individuals per parents.
    num_individuals_per_parents: usize,
    // phantom types
}

impl MaximizeSelector {
    /// Constructs a new instance of the `MaximizeSelector`.
    pub fn new(truncation_threshold: f64, num_individuals_per_parents: usize) -> Self {
        MaximizeSelector {
            truncation_threshold: truncation_threshold,
            num_individuals_per_parents: num_individuals_per_parents,
        }
    }

    /// Returns the truncation threshold used by this `MaximizeSelector`.
    ///
    /// The truncation threshold is the ratio between the number of parents
    /// to be selected and the size of the population:
    /// threshold = number of parents / size of population
    pub fn truncation_threshold(&self) -> f64 {
        self.truncation_threshold
    }

    /// Sets the truncation threshold to the given value.
    ///
    /// The truncation threshold is the ratio between the number of parents
    /// to be selected and the size of the population:
    /// threshold = number of parents / size of population
    pub fn set_truncation_threshold(&mut self, value: f64) {
        self.truncation_threshold = value;
    }

    /// Returns the number of individuals per parents use by this selector.
    pub fn num_individuals_per_parents(&self) -> usize {
        self.num_individuals_per_parents
    }

    /// Sets the number of individuals per parents to the given value.
    pub fn set_num_individuals_per_parents(&mut self, value: usize) {
        self.num_individuals_per_parents = value;
    }
}

/// Can be used for single-objective optimization
impl SingleObjective for MaximizeSelector {}
/// Can be used for multi-objective optimization
impl MultiObjective for MaximizeSelector {}

impl GeneticOperator for MaximizeSelector {
    fn name() -> String {
        "Maximizing-Truncation-Selection".to_string()
    }
}

impl<G, F> SelectionOp<G, F> for MaximizeSelector
    where G: Genotype, F: Fitness
{
    fn select_from(&self, evaluated: &EvaluatedPopulation<G, F>) -> Result<Vec<Parents<G>>, SimError> {
        let individuals = evaluated.individuals();
        let fitness_values = evaluated.fitness_values();

        // mating pool holds indices to the individuals and fitness_values slices
        let mut mating_pool: Vec<usize> = (0..fitness_values.len()).collect();
        // sort mating pool from best performing to worst performing index
        mating_pool.sort_by(|x, y| fitness_values[*y].cmp(&fitness_values[*x]));
        let mating_pool = mating_pool;

        let num_parents_to_select = (individuals.len() as f64 * self.truncation_threshold).floor() as usize;
        let pool_size = mating_pool.len();
        let mut selected: Vec<Parents<G>> = Vec::with_capacity(num_parents_to_select);

        let mut index_m = 0;
        for _ in 0..num_parents_to_select {
            let mut tuple = Vec::with_capacity(self.num_individuals_per_parents);
            for _ in 0..self.num_individuals_per_parents {
                // index into mating pool
                index_m = index_m % pool_size;
                // index into individuals slice
                let index_i = mating_pool[index_m];
                tuple.push(individuals[index_i].clone());
                index_m += 1;
            }
            selected.push(tuple);
        }
        Ok(selected)
    }
}
