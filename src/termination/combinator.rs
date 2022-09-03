use crate::{
    algorithm::Algorithm,
    simulation::State,
    termination::{StopFlag, Termination}, genetic::Genotype,
};
use std::marker::PhantomData;

// TODO add doc comments
pub fn and<T1, T2, A, G>(condition1: T1, condition2: T2) -> And<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    And::new(condition1, condition2)
}

// TODO add doc comments
#[derive(Clone, Debug, PartialEq)]
pub struct And<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    condition1: T1,
    condition2: T2,
    _a: PhantomData<A>,
    _phantom_data: PhantomData<G>
}

impl<T1, T2, A, G> And<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    pub fn new(condition1: T1, condition2: T2) -> Self {
        And {
            condition1,
            condition2,
            _a: PhantomData,
            _phantom_data: PhantomData
        }
    }

    pub fn condition1(&self) -> &T1 {
        &self.condition1
    }

    pub fn condition2(&self) -> &T2 {
        &self.condition2
    }
}

impl<T1, T2, A, G> Termination<A, G> for And<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    fn evaluate(&mut self, state: &State<A, G>) -> StopFlag {
        let mut reasons = Vec::with_capacity(2);
        match self.condition1.evaluate(state) {
            StopFlag::StopNow(reason) => reasons.push(reason),
            StopFlag::Continue => (),
        }
        match self.condition2.evaluate(state) {
            StopFlag::StopNow(reason) => reasons.push(reason),
            StopFlag::Continue => (),
        }
        match reasons.len() {
            0 | 1 => StopFlag::Continue,
            _ => StopFlag::StopNow(reasons.join(" and ")), /* TODO how combine the two `StopReason`s preserving combinator semantics? */
        }
    }
}

// TODO add doc comments
pub fn or<T1, T2, A, G>(condition1: T1, condition2: T2) -> Or<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    Or::new(condition1, condition2)
}

// TODO add doc comments
#[derive(Clone, Debug, PartialEq)]
pub struct Or<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    condition1: T1,
    condition2: T2,
    _a: PhantomData<A>,
    _phantom_data: PhantomData<G>
}

impl<T1, T2, A, G> Or<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    pub fn new(condition1: T1, condition2: T2) -> Self {
        Or {
            condition1,
            condition2,
            _a: PhantomData,
            _phantom_data: PhantomData
        }
    }

    pub fn condition1(&self) -> &T1 {
        &self.condition1
    }

    pub fn condition2(&self) -> &T2 {
        &self.condition2
    }
}

impl<T1, T2, A, G> Termination<A, G> for Or<T1, T2, A, G>
where
    T1: Termination<A, G>,
    T2: Termination<A, G>,
    A: Algorithm<G>,
    G: Genotype
{
    fn evaluate(&mut self, state: &State<A, G>) -> StopFlag {
        let mut reasons = Vec::with_capacity(2);
        match self.condition1.evaluate(state) {
            StopFlag::StopNow(reason) => reasons.push(reason),
            StopFlag::Continue => (),
        }
        match self.condition2.evaluate(state) {
            StopFlag::StopNow(reason) => reasons.push(reason),
            StopFlag::Continue => (),
        }
        match reasons.len() {
            0 => StopFlag::Continue,
            1 => StopFlag::StopNow(reasons[0].clone()),
            _ => StopFlag::StopNow(reasons.join(" and ")), /* TODO how combine the two `StopReason`s preserving combinator semantics? */
        }
    }
}
