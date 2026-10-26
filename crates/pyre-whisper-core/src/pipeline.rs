use crate::WhisburnResult;

/// A single pure transformation step in the speech pipeline.
pub trait PipelineStep<I, O> {
    fn name(&self) -> &'static str;
    fn apply(&self, input: I) -> WhisburnResult<O>;
}

/// Compose two pipeline steps into a single step (functional composition).
pub fn compose<A, B, C, S1, S2>(first: S1, second: S2) -> ComposedStep<A, B, C, S1, S2>
where
    S1: PipelineStep<A, B>,
    S2: PipelineStep<B, C>,
{
    ComposedStep {
        first,
        second,
        _phantom: std::marker::PhantomData,
    }
}

pub struct ComposedStep<A, B, C, S1, S2>
where
    S1: PipelineStep<A, B>,
    S2: PipelineStep<B, C>,
{
    first: S1,
    second: S2,
    _phantom: std::marker::PhantomData<(A, B, C)>,
}

impl<A, B, C, S1, S2> PipelineStep<A, C> for ComposedStep<A, B, C, S1, S2>
where
    S1: PipelineStep<A, B>,
    S2: PipelineStep<B, C>,
{
    fn name(&self) -> &'static str {
        "composed"
    }

    fn apply(&self, input: A) -> WhisburnResult<C> {
        let mid = self.first.apply(input)?;
        self.second.apply(mid)
    }
}

/// Linear pipeline metadata container.
pub struct Pipeline {
    step_names: Vec<&'static str>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            step_names: Vec::new(),
        }
    }

    pub fn register_step(mut self, name: &'static str) -> Self {
        self.step_names.push(name);
        self
    }

    pub fn step_count(&self) -> usize {
        self.step_names.len()
    }

    pub fn step_names(&self) -> &[&'static str] {
        &self.step_names
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}