use core::cmp::Ordering;

use crate::{SpaceByTimeMachine, snapshot::Snapshot};

// cmk000 consider boxing, etc to make smaller
pub enum Message0 {
    Snapshot {
        part_index: usize,
        snapshot: Snapshot,
    },
    SpaceByTimeMachine {
        part_index: usize,
        space_by_time_machine: SpaceByTimeMachine,
    },
}

impl Message0 {
    pub const fn part_index(&self) -> usize {
        match self {
            Self::SpaceByTimeMachine { part_index, .. } | Self::Snapshot { part_index, .. } => {
                *part_index
            }
        }
    }
}

// Implement PartialEq to define equality comparison
#[allow(clippy::missing_trait_methods)]
impl PartialEq for Message0 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // Two Snapshots are equal if part_index and step_index() match
            (
                Self::Snapshot {
                    part_index: p1,
                    snapshot: s1,
                },
                Self::Snapshot {
                    part_index: p2,
                    snapshot: s2,
                },
            ) => p1 == p2 && s1.space_by_time.step_index() == s2.space_by_time.step_index(),

            // Two SpaceByTimeMachine variants are equal if their part_index matches
            (
                Self::SpaceByTimeMachine { part_index: p1, .. },
                Self::SpaceByTimeMachine { part_index: p2, .. },
            ) => p1 == p2,

            // Snapshot and SpaceByTimeMachine should never be equal
            _ => false,
        }
    }
}

// Implement Eq (marker trait that requires PartialEq)
#[allow(clippy::missing_trait_methods)]
impl Eq for Message0 {}

// Implement Ord for BinaryHeap prioritization
#[allow(clippy::missing_trait_methods)]
impl Ord for Message0 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare part_index (smaller is higher priority)
        match self.part_index().cmp(&other.part_index()) {
            Ordering::Less => Ordering::Greater, // Reverse because BinaryHeap is max-heap
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => match (self, other) {
                // Snapshot is higher priority than SpaceByTimeMachine
                (Self::Snapshot { .. }, Self::SpaceByTimeMachine { .. }) => Ordering::Greater,
                (Self::SpaceByTimeMachine { .. }, Self::Snapshot { .. }) => Ordering::Less,
                // If both are Snapshot, compare step_index()
                (Self::Snapshot { snapshot: s1, .. }, Self::Snapshot { snapshot: s2, .. }) => {
                    s2.space_by_time
                        .step_index()
                        .cmp(&s1.space_by_time.step_index()) // Lower step_index is higher priority
                }
                // If both are SpaceByTimeMachine, they must have different part_index (asserted)
                (Self::SpaceByTimeMachine { .. }, Self::SpaceByTimeMachine { .. }) => {
                    Ordering::Equal
                }
            },
        }
    }
}

#[allow(clippy::missing_trait_methods)]
impl PartialOrd for Message0 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other)) // Delegate to `Ord`
    }
}
