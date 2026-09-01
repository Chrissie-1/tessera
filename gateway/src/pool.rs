//! Worker pool: connection management, least-in-flight routing, load shedding.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tonic::transport::{Channel, Endpoint};

use crate::pb::inference_client::InferenceClient;

/// A single worker and this gateway's view of how busy it is.
struct Worker {
    endpoint: String,
    client: InferenceClient<Channel>,
    in_flight: AtomicUsize,
}

/// Borrowed worker slot. Decrements the in-flight counter when dropped, so a
/// panicking or early-returning handler cannot leak capacity.
pub struct Lease {
    pool: Arc<WorkerPool>,
    /// Index of the worker this lease reserves capacity on.
    index: usize,
    pub client: InferenceClient<Channel>,
    pub endpoint: String,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.workers[self.index]
            .in_flight
            .fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct WorkerPool {
    workers: Vec<Worker>,
    /// Requests one worker may execute concurrently before the gateway sheds load.
    max_in_flight: usize,
}

impl WorkerPool {
    /// Connect lazily to every endpoint.
    ///
    /// Lazy connection is deliberate: the gateway must start even when workers
    /// are still loading their model weights, which takes far longer than the
    /// gateway's own startup.
    pub fn connect(
        endpoints: &[String],
        max_in_flight: usize,
    ) -> Result<Self, tonic::transport::Error> {
        let mut workers = Vec::with_capacity(endpoints.len());
        for endpoint in endpoints {
            let channel = Endpoint::from_shared(endpoint.clone())?
                .connect_timeout(std::time::Duration::from_secs(5))
                .timeout(std::time::Duration::from_secs(300))
                .connect_lazy();
            workers.push(Worker {
                endpoint: endpoint.clone(),
                client: InferenceClient::new(channel),
                in_flight: AtomicUsize::new(0),
            });
        }
        Ok(Self {
            workers,
            max_in_flight,
        })
    }

    pub fn endpoints(&self) -> Vec<String> {
        self.workers.iter().map(|w| w.endpoint.clone()).collect()
    }

    /// Clone the client for a specific worker without reserving capacity.
    ///
    /// Health probes must not consume a load-shedding slot, and must address
    /// one named worker rather than whichever is least loaded.
    pub fn client_at(&self, index: usize) -> Option<InferenceClient<Channel>> {
        self.workers.get(index).map(|w| w.client.clone())
    }

    pub fn in_flight(&self) -> Vec<usize> {
        self.workers
            .iter()
            .map(|w| w.in_flight.load(Ordering::SeqCst))
            .collect()
    }

    /// Reserve the least-loaded worker, or `None` when every worker is saturated.
    ///
    /// The counter is claimed with a compare-and-swap so two concurrent requests
    /// cannot both take the last slot on the same worker.
    pub fn acquire(self: &Arc<Self>) -> Option<Lease> {
        loop {
            let (index, observed) = self
                .workers
                .iter()
                .enumerate()
                .map(|(i, w)| (i, w.in_flight.load(Ordering::SeqCst)))
                .min_by_key(|(_, load)| *load)?;

            if observed >= self.max_in_flight {
                return None;
            }

            if self.workers[index]
                .in_flight
                .compare_exchange(observed, observed + 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Some(Lease {
                    pool: Arc::clone(self),
                    index,
                    client: self.workers[index].client.clone(),
                    endpoint: self.workers[index].endpoint.clone(),
                });
            }
            // Lost the race; re-read and pick again.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(n: usize, max_in_flight: usize) -> Arc<WorkerPool> {
        // connect_lazy never dials, so these endpoints need not be listening.
        let endpoints: Vec<String> = (0..n)
            .map(|i| format!("http://127.0.0.1:{}", 50051 + i))
            .collect();
        Arc::new(WorkerPool::connect(&endpoints, max_in_flight).expect("pool connects"))
    }

    #[tokio::test]
    async fn acquire_spreads_load_across_workers() {
        let pool = pool(3, 8);
        let leases: Vec<Lease> = (0..3).map(|_| pool.acquire().expect("capacity")).collect();

        let picked: Vec<usize> = leases.iter().map(|l| l.index).collect();
        assert_eq!(pool.in_flight(), vec![1, 1, 1]);
        // Each of the three workers was chosen exactly once.
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn acquire_prefers_the_least_loaded_worker() {
        let pool = pool(2, 8);
        let first = pool.acquire().expect("capacity");
        let second = pool.acquire().expect("capacity");
        assert_ne!(first.index, second.index);

        drop(first);
        // Worker `first.index` is now idle, so it must win the next round.
        let third = pool.acquire().expect("capacity");
        assert_eq!(pool.in_flight()[third.index], 1);
        assert_ne!(third.index, second.index);
    }

    #[tokio::test]
    async fn lease_release_is_automatic_on_drop() {
        let pool = pool(1, 4);
        {
            let _lease = pool.acquire().expect("capacity");
            assert_eq!(pool.in_flight(), vec![1]);
        }
        assert_eq!(pool.in_flight(), vec![0]);
    }

    #[tokio::test]
    async fn saturated_pool_sheds_load() {
        let pool = pool(2, 1);
        let _a = pool.acquire().expect("capacity");
        let _b = pool.acquire().expect("capacity");

        assert_eq!(pool.in_flight(), vec![1, 1]);
        assert!(pool.acquire().is_none(), "expected the pool to shed load");
    }

    #[tokio::test]
    async fn capacity_returns_after_shedding() {
        let pool = pool(1, 1);
        let lease = pool.acquire().expect("capacity");
        assert!(pool.acquire().is_none());

        drop(lease);
        assert!(pool.acquire().is_some(), "capacity should be reusable");
    }

    #[tokio::test]
    async fn client_at_does_not_reserve_capacity() {
        let pool = pool(2, 1);
        assert!(pool.client_at(0).is_some());
        assert!(pool.client_at(5).is_none());
        assert_eq!(pool.in_flight(), vec![0, 0]);
    }
}
