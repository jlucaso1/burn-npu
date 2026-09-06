//! Bounded single-flight cache. Native compilation and native object destruction
//! never run under the map lock. Failed builds have a retry interval, and entries
//! under construction cannot be evicted by normal capacity pressure.
use super::OpenVinoUnavailable;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

type BuildResult<V> = (Result<Arc<V>, OpenVinoUnavailable>, Instant);
struct Entry<V> {
    value: OnceLock<BuildResult<V>>,
    charge: usize,
}
struct State<K, V> {
    entries: HashMap<K, Arc<Entry<V>>>,
    order: VecDeque<K>,
    stats: CacheStats,
}
/// Cache accounting counts source weights and expected I/O bytes, not opaque
/// compiler/driver allocations. Evicted entries may remain alive in active calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub estimated_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub build_failures: u64,
    pub evictions: u64,
    pub admission_failures: u64,
}
pub(super) struct BuildCache<K, V> {
    state: Mutex<State<K, V>>,
    max_entries: usize,
    max_bytes: usize,
    retry_after: Duration,
}
impl<K: Eq + Hash + Clone, V> BuildCache<K, V> {
    pub(super) fn new(max_entries: usize, max_bytes: usize, retry_after: Duration) -> Self {
        Self {
            state: Mutex::new(State {
                entries: HashMap::new(),
                order: VecDeque::new(),
                stats: CacheStats::default(),
            }),
            max_entries,
            max_bytes,
            retry_after,
        }
    }
    pub(super) fn get_or_try_init(
        &self,
        key: K,
        charge: usize,
        build: impl FnOnce() -> Result<V, OpenVinoUnavailable>,
    ) -> Result<Arc<V>, OpenVinoUnavailable> {
        // Keep retired native objects alive until after releasing the map lock.
        let mut retired = Vec::new();
        let mut state = self.state.lock().map_err(|_| OpenVinoUnavailable)?;
        if charge > self.max_bytes || self.max_entries == 0 {
            state.stats.admission_failures += 1;
            return Err(OpenVinoUnavailable);
        }
        let expired = state
            .entries
            .get(&key)
            .and_then(|entry| entry.value.get())
            .is_some_and(|(result, finished)| {
                result.is_err() && finished.elapsed() >= self.retry_after
            });
        if expired {
            let old = state.entries.remove(&key).expect("entry checked above");
            state.stats.estimated_bytes -= old.charge;
            state.stats.entries = state.entries.len();
            state.order.retain(|k| k != &key);
            retired.push(old);
        }
        let slot = if let Some(entry) = state.entries.get(&key).cloned() {
            state.stats.hits += 1;
            state.order.retain(|k| k != &key);
            state.order.push_back(key);
            entry
        } else {
            while state.entries.len() >= self.max_entries
                || state.stats.estimated_bytes > self.max_bytes - charge
            {
                let Some(index) = state
                    .order
                    .iter()
                    .position(|k| state.entries[k].value.get().is_some())
                else {
                    state.stats.admission_failures += 1;
                    return Err(OpenVinoUnavailable);
                };
                let oldest = state.order.remove(index).expect("valid index");
                let entry = state.entries.remove(&oldest).expect("LRU and map agree");
                state.stats.estimated_bytes -= entry.charge;
                state.stats.entries = state.entries.len();
                state.stats.evictions += 1;
                retired.push(entry);
            }
            let slot = Arc::new(Entry {
                value: OnceLock::new(),
                charge,
            });
            state.entries.insert(key.clone(), Arc::clone(&slot));
            state.order.push_back(key);
            state.stats.estimated_bytes += charge;
            state.stats.misses += 1;
            slot
        };
        state.stats.entries = state.entries.len();
        drop(state);
        drop_retired(retired);
        let (result, _) = slot.value.get_or_init(|| {
            let result = build().map(Arc::new);
            if result.is_err() {
                if let Ok(mut state) = self.state.lock() {
                    state.stats.build_failures += 1;
                }
            }
            (result, Instant::now())
        });
        result.clone()
    }
    /// Quarantine a failed executor, but never overwrite a newer replacement.
    pub(super) fn mark_failed(&self, key: &K, expected: &Arc<V>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(old) = state.entries.get(key) else {
            return;
        };
        if !old.value.get().is_some_and(|(result, _)| {
            result
                .as_ref()
                .is_ok_and(|current| Arc::ptr_eq(current, expected))
        }) {
            return;
        }
        let replacement = Arc::new(Entry {
            charge: old.charge,
            value: OnceLock::from((Err(OpenVinoUnavailable), Instant::now())),
        });
        let retired = state.entries.insert(key.clone(), replacement);
        drop(state);
        drop_retired(retired);
    }
    pub(super) fn stats(&self) -> CacheStats {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stats
    }
    pub(super) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let retired = std::mem::take(&mut state.entries);
        state.order.clear();
        state.stats.entries = 0;
        state.stats.estimated_bytes = 0;
        drop(state);
        self.state.clear_poison();
        drop_retired(retired.into_values());
    }
}

/// Drop retired entries with the native runtime loaded on this thread.
///
/// Native destructors run against thread-local runtime tables, which a fresh
/// worker may never have loaded. If loading fails, leak rather than run
/// destructors without the runtime.
fn drop_retired<V>(retired: impl IntoIterator<Item = Arc<Entry<V>>>) {
    let mut retired = retired.into_iter().peekable();
    if retired.peek().is_some() && super::load_openvino().is_err() {
        retired.for_each(std::mem::forget);
        return;
    }
    drop(retired);
}

/// A cached native executor that can be quarantined after a failed call.
pub(super) trait CachedEntry {
    fn has_failed(&self) -> bool;
    fn set_failed(&mut self);
}

/// Look up (or build), lock, run and quarantine-on-error in one place.
///
/// Callers supply only input filling and output reading; the lock discipline
/// and failure protocol stay in lockstep across graphs.
pub(super) fn run_cached<K, E>(
    cache: &BuildCache<K, Mutex<E>>,
    key: K,
    charge: usize,
    build: impl FnOnce() -> Result<E, OpenVinoUnavailable>,
    run: impl FnOnce(&mut E) -> Result<Vec<f32>, OpenVinoUnavailable>,
) -> Result<Vec<f32>, OpenVinoUnavailable>
where
    K: Eq + Hash + Clone,
    E: CachedEntry,
{
    let cached = cache.get_or_try_init(key.clone(), charge, || build().map(Mutex::new))?;
    let mut entry = cached.lock().map_err(|_| OpenVinoUnavailable)?;
    if entry.has_failed() {
        return Err(OpenVinoUnavailable);
    }
    let outcome = run(&mut entry);
    if outcome.is_err() {
        entry.set_failed();
        drop(entry);
        cache.mark_failed(&key, &cached);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };
    fn cache() -> BuildCache<usize, usize> {
        BuildCache::new(2, 8, Duration::from_secs(60))
    }
    #[test]
    fn lru_byte_budget_and_live_values() {
        let cache = cache();
        let old = cache.get_or_try_init(1, 4, || Ok(10)).unwrap();
        cache.get_or_try_init(2, 4, || Ok(20)).unwrap();
        cache.get_or_try_init(1, 4, || panic!("cache hit")).unwrap();
        cache.get_or_try_init(3, 4, || Ok(30)).unwrap();
        assert_eq!(*old, 10);
        assert!(!cache.state.lock().unwrap().entries.contains_key(&2));
        assert_eq!(cache.stats().estimated_bytes, 8);
        assert_eq!(cache.stats().evictions, 1);
        cache.clear();
        assert_eq!(cache.stats().estimated_bytes, 0);
        assert_eq!(*old, 10);
    }
    #[test]
    fn failed_builds_back_off_and_oversized_builds_never_start() {
        let cache = cache();
        assert!(cache
            .get_or_try_init(1, 4, || Err(OpenVinoUnavailable))
            .is_err());
        assert!(cache
            .get_or_try_init(1, 4, || panic!("retry storm"))
            .is_err());
        assert!(cache
            .get_or_try_init(2, 9, || panic!("over budget"))
            .is_err());
        assert_eq!(cache.stats().build_failures, 1);
        let immediate = BuildCache::new(1, 4, Duration::ZERO);
        assert!(immediate
            .get_or_try_init(1, 4, || Err(OpenVinoUnavailable))
            .is_err());
        assert_eq!(*immediate.get_or_try_init(1, 4, || Ok(7)).unwrap(), 7);
    }
    #[test]
    fn one_builder_for_many_callers() {
        let cache = cache();
        let count = AtomicUsize::new(0);
        let start = Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    start.wait();
                    assert_eq!(
                        *cache
                            .get_or_try_init(1, 4, || {
                                count.fetch_add(1, Ordering::SeqCst);
                                Ok(42)
                            })
                            .unwrap(),
                        42
                    );
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn unrelated_builds_do_not_hold_the_global_lock() {
        let cache = cache();
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| {
            for key in 0..2 {
                let barrier = &barrier;
                let cache = &cache;
                scope.spawn(move || {
                    cache
                        .get_or_try_init(key, 4, || {
                            barrier.wait();
                            Ok(key)
                        })
                        .unwrap()
                });
            }
        });
    }
    #[test]
    fn failed_executor_is_quarantined_without_invalidating_replacements() {
        let cache = cache();
        let old = cache.get_or_try_init(1, 4, || Ok(10)).unwrap();
        cache.mark_failed(&1, &old);
        assert!(cache
            .get_or_try_init(1, 4, || panic!("quarantine bypassed"))
            .is_err());
        cache.clear();
        let new = cache.get_or_try_init(1, 4, || Ok(20)).unwrap();
        cache.mark_failed(&1, &old);
        assert_eq!(
            *cache
                .get_or_try_init(1, 4, || panic!("new value invalidated"))
                .unwrap(),
            *new
        );
    }
}
