//! The `statistic` module provides functionality to collect and display
//! statistic about a genetic algorithm application and its execution.

use crate::types::fmt::Display;
use chrono::{Duration, Local};
use std::{
    convert::From,
    fmt,
    ops::{Add, AddAssign},
};

