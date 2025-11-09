# Memory Segmentation with RL in ASearcher: Validation Plan

## Executive Summary

This document outlines a concrete plan to validate the **concept-coherent memory segmentation with RL** approach within the ASearcher framework. ASearcher is an ideal testbed because:

1. ✅ Has clear **agent task success metrics** (answer correctness)
2. ✅ Faces **long-context memory challenges** (32k+ tokens)
3. ✅ Requires **multi-hop reasoning** across memory segments
4. ✅ Has existing **RL infrastructure** (GRPO training pipeline)
5. ✅ Natural **segmentation boundaries** already exist (Records)

---

## Phase 1: Memory Segment Selection Policy (Weeks 1-2)

### Goal
Train an RL policy to **select which memory segments to include in context**, maximizing agent task success while staying within token limits.

### Architecture

```python
# New component to add to ASearcher
class MemorySegmentSelector:
    """
    RL-based policy that decides which historical memory segments
    to include when preparing the next prompt.
    """

    def __init__(self, embed_dim=1024, hidden_dim=512):
        # State encoder: Encodes each Record into a concept vector
        self.record_encoder = nn.TransformerEncoder(...)

        # Selection policy: Outputs importance score for each segment
        self.selection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Importance score [0, 1]
        )

    def forward(self, memory_records, current_state, budget=24000):
        """
        Args:
            memory_records: List[Record] - all historical records
            current_state: str - current reasoning state / question
            budget: int - max tokens allowed in context

        Returns:
            selected_indices: List[int] - indices of records to include
            importance_scores: Tensor - importance of each record
        """
        # 1. Encode all records into concept vectors
        record_embeddings = self.encode_records(memory_records)

        # 2. Encode current state
        state_embedding = self.encode_state(current_state)

        # 3. Compute relevance of each record to current state
        relevance = cosine_similarity(record_embeddings, state_embedding)

        # 4. Compute importance scores (RL policy)
        importance_scores = self.selection_head(
            torch.cat([record_embeddings, relevance.unsqueeze(-1)], dim=-1)
        )

        # 5. Select top-k records within budget
        selected_indices = self.select_within_budget(
            importance_scores, memory_records, budget
        )

        return selected_indices, importance_scores
```

### Integration with AgentMemory

**Modify `/home/user/ASearcher/agent/asearcher.py:23-67`:**

```python
class AgentMemory:
    def __init__(self, prompt, use_rl_selection=False):
        self.memory = [Record(type="prompt", text=prompt)]
        self.use_rl_selection = use_rl_selection
        self.selector = MemorySegmentSelector() if use_rl_selection else None

        # Track segment metadata for RL
        self.segment_importance_history = []  # Store importance scores

    def prepare_prompt(self, max_length=25000, current_state=None):
        """Modified to use RL-based selection when enabled"""

        if self.use_rl_selection and self.selector is not None:
            # RL-based selection
            selected_indices, importance_scores = self.selector(
                memory_records=self.memory,
                current_state=current_state,
                budget=max_length
            )

            # Store for reward computation
            self.segment_importance_history.append({
                'scores': importance_scores,
                'selected': selected_indices,
                'turn': self.llm_gen_count()
            })

            # Build prompt from selected segments
            selected_records = [self.memory[i] for i in selected_indices]
            prompt = self._format_records(selected_records)

        else:
            # Original sliding window approach
            prompt = self._original_prepare_prompt(max_length)

        return prompt
```

### Reward Design

**Primary Reward: Task Success (weight = 1.0)**

```python
def compute_segment_selection_reward(episode_data):
    """
    Compute reward for memory segment selection policy.

    Args:
        episode_data: Dict containing:
            - final_answer: str
            - ground_truth: str
            - segment_selection_history: List[Dict] with importance scores

    Returns:
        total_reward: float
        segment_level_rewards: List[float]
    """

    # 1. PRIMARY: Task success (same as current ASearcher reward)
    task_success = llm_judge_equivalence(
        episode_data['final_answer'],
        episode_data['ground_truth']
    )  # Returns 0.0 or 1.0

    # 2. AUXILIARY: Credit assignment to segments
    #    Which segments contributed to success/failure?
    segment_contributions = assign_credit_to_segments(
        episode_data['segment_selection_history'],
        task_success
    )

    # 3. AUXILIARY: Efficiency bonus (use fewer tokens)
    token_efficiency = 1.0 - (tokens_used / max_tokens)

    # 4. Combined reward
    total_reward = (
        task_success +                    # Primary: 1.0 weight
        0.1 * segment_contributions +     # Auxiliary: 0.1 weight
        0.05 * token_efficiency          # Auxiliary: 0.05 weight
    )

    return total_reward, segment_contributions
```

**Credit Assignment Strategy:**

```python
def assign_credit_to_segments(selection_history, task_success):
    """
    Assign credit to segments based on:
    1. How often they were selected
    2. At what critical moments they were selected
    3. Their importance scores
    """

    segment_credits = defaultdict(float)

    for turn_data in selection_history:
        importance_scores = turn_data['scores']
        selected_indices = turn_data['selected']
        turn = turn_data['turn']

        # Weight later turns more (closer to answer)
        turn_weight = (turn + 1) / len(selection_history)

        for idx in selected_indices:
            # Segments selected in critical moments get more credit
            segment_credits[idx] += (
                importance_scores[idx] *
                turn_weight *
                task_success  # Only get credit if task succeeded
            )

    return segment_credits
```

### Training Loop Modification

**Modify `/home/user/ASearcher/ASearcher/train/asearcher_reasoning.py`:**

```python
async def arun_episode_with_memory_selection(
    question_data,
    model_engine,
    memory_selector,
    **kwargs
):
    """
    Modified episode runner that uses RL-based memory selection.
    """

    # Initialize memory with RL selector
    memory = AgentMemory(
        prompt=question_data['question'],
        use_rl_selection=True
    )
    memory.selector = memory_selector  # Use trainable selector

    # Run episode as normal, but memory.prepare_prompt() now uses RL
    trajectory_data = []

    for turn in range(max_turns):
        # 1. Prepare prompt using RL-based segment selection
        current_state = question_data['question']  # Could be more sophisticated
        prompt = memory.prepare_prompt(
            max_length=24000,
            current_state=current_state
        )

        # 2. Generate next action
        llm_response = await model_engine.generate(prompt, **gen_config)

        # 3. Execute action (search/access/answer)
        action_result = execute_action(llm_response)

        # 4. Store in memory
        memory.add_llm_gen(llm_response, rl_metadata=...)
        if action_result:
            memory.add_record(action_result)

        # 5. Check for answer
        if is_answer(llm_response):
            final_answer = extract_answer(llm_response)
            break

        trajectory_data.append({
            'prompt': prompt,
            'response': llm_response,
            'action': action_result,
            'segment_selections': memory.segment_importance_history[-1]
        })

    # 6. Compute reward (task success + segment-level credit)
    reward, segment_credits = compute_segment_selection_reward({
        'final_answer': final_answer,
        'ground_truth': question_data['answer'],
        'segment_selection_history': memory.segment_importance_history
    })

    return {
        'trajectory': trajectory_data,
        'reward': reward,
        'segment_credits': segment_credits,
        'final_answer': final_answer
    }
```

### Training Update

```python
def update_memory_selector(memory_selector, trajectories, optimizer):
    """
    Update the memory segment selector using policy gradient.
    """

    # 1. Collect all segment selection decisions
    all_log_probs = []
    all_rewards = []

    for traj in trajectories:
        for step in traj['trajectory']:
            selection_data = step['segment_selections']

            # Log probability of selecting these segments
            log_prob = compute_selection_log_prob(
                importance_scores=selection_data['scores'],
                selected_indices=selection_data['selected']
            )

            all_log_probs.append(log_prob)
            all_rewards.append(traj['reward'])  # Could use step-level credit

    # 2. Compute advantages (group normalization like GRPO)
    rewards_tensor = torch.tensor(all_rewards)
    advantages = (rewards_tensor - rewards_tensor.mean())
    if rewards_tensor.std() > 1e-3:
        advantages = advantages / rewards_tensor.std()

    # 3. Policy gradient loss
    log_probs_tensor = torch.stack(all_log_probs)
    loss = -(log_probs_tensor * advantages).mean()

    # 4. Update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(memory_selector.parameters(), 1.0)
    optimizer.step()

    return loss.item()
```

### Expected Outcomes

**Success Metrics:**

1. **Task Success Rate ↑**: Answer accuracy should improve compared to baseline
   - Baseline: Current sliding window approach
   - Target: +5-10% improvement on GAIA/xBench

2. **Context Efficiency ↑**: Use fewer tokens while maintaining accuracy
   - Baseline: ~24k tokens average
   - Target: ~18k tokens average (25% reduction)

3. **Segment Importance Learned**: Policy should learn to prioritize:
   - Recent search results > old search results
   - Webpages containing answer keywords > irrelevant pages
   - Critical reasoning steps > intermediate attempts

4. **Credit Assignment Works**: Segment-level credits should correlate with:
   - Ground truth answer appearance in segment
   - Semantic relevance to question
   - Temporal proximity to successful answer

**Validation Tests:**

```python
# Test 1: Does policy select relevant segments?
def test_segment_relevance():
    """
    For questions with known answer locations, check if policy
    selects segments containing those answers.
    """
    relevant_segment_recall = []

    for question, answer_location in test_set:
        selected_segments = memory_selector.select(...)
        if answer_location in selected_segments:
            relevant_segment_recall.append(1.0)
        else:
            relevant_segment_recall.append(0.0)

    print(f"Relevant Segment Recall: {np.mean(relevant_segment_recall)}")

# Test 2: Does policy improve efficiency?
def test_token_efficiency():
    """Compare tokens used vs. baseline while maintaining accuracy"""
    baseline_tokens = run_with_sliding_window(test_set)
    rl_tokens = run_with_rl_selection(test_set)

    print(f"Token Reduction: {(1 - rl_tokens/baseline_tokens) * 100}%")

# Test 3: Does policy handle multi-hop reasoning?
def test_multihop_reasoning():
    """
    For multi-hop questions, check if policy selects ALL necessary segments.
    """
    multihop_success_rate = []

    for question in multihop_test_set:
        required_segments = question.get_required_segments()
        selected_segments = memory_selector.select(...)

        coverage = len(set(required_segments) & set(selected_segments)) / len(required_segments)
        multihop_success_rate.append(coverage)

    print(f"Multi-hop Coverage: {np.mean(multihop_success_rate)}")
```

---

## Phase 2: Dynamic Memory Segmentation (Weeks 3-4)

### Goal
Go beyond selection—train policy to **dynamically create segment boundaries** in incoming information (search results, webpages).

### Architecture

```python
class DynamicMemorySegmenter:
    """
    Policy that decides where to insert boundaries in continuous text.
    Similar to your original SONAR-based approach but adapted for ASearcher.
    """

    def __init__(self):
        # Sentence encoder (could use SONAR or simpler embeddings)
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Boundary prediction policy
        self.boundary_predictor = nn.LSTM(
            input_size=384 + 10,  # embedding + features
            hidden_size=256,
            num_layers=2
        )
        self.boundary_head = nn.Linear(256, 2)  # continue vs boundary

    def segment_text(self, text, context=None):
        """
        Dynamically segment incoming text (e.g., a webpage).

        Args:
            text: str - raw text to segment
            context: str - current question/reasoning context

        Returns:
            segments: List[str] - segmented chunks
            boundary_decisions: List[int] - where boundaries were placed
        """
        # 1. Split into sentences
        sentences = sent_tokenize(text)

        # 2. Encode sentences
        embeddings = self.sentence_encoder.encode(sentences)

        # 3. Compute features for each position
        features = []
        for i in range(len(sentences)):
            feat = {
                'embedding': embeddings[i],
                'coherence_forward': cosine_similarity(embeddings[i], embeddings[i+1]) if i < len(sentences)-1 else 0,
                'coherence_backward': cosine_similarity(embeddings[i], embeddings[i-1]) if i > 0 else 0,
                'sentence_length': len(sentences[i]),
                'position': i / len(sentences),
                # ... other features
            }
            features.append(feat)

        # 4. Policy decides boundaries
        boundary_decisions = []
        current_segment = []

        for i, feat_dict in enumerate(features):
            # Prepare state
            state = torch.cat([
                torch.tensor(feat_dict['embedding']),
                torch.tensor([
                    feat_dict['coherence_forward'],
                    feat_dict['coherence_backward'],
                    feat_dict['position'],
                    # ... other features
                ])
            ])

            # Policy decision
            logits = self.boundary_head(self.boundary_predictor(state))
            action = torch.argmax(logits)  # 0: continue, 1: boundary

            current_segment.append(sentences[i])
            boundary_decisions.append(action.item())

            if action == 1:  # Insert boundary
                segments.append(' '.join(current_segment))
                current_segment = []

        # Add final segment
        if current_segment:
            segments.append(' '.join(current_segment))

        return segments, boundary_decisions
```

### Integration Point: Webpage Processing

**Modify webpage processing in SearchToolBox:**

```python
# In /home/user/ASearcher/ASearcher/utils/search_tool.py

async def access_webpage_with_dynamic_segmentation(url, segmenter=None):
    """
    Fetch webpage and use RL policy to segment it into coherent chunks.
    """

    # 1. Fetch raw content
    raw_text = await fetch_webpage(url)

    # 2. Use RL-based segmentation (if available)
    if segmenter is not None:
        segments, boundary_decisions = segmenter.segment_text(
            text=raw_text,
            context=current_question  # Pass question for relevance
        )
    else:
        # Fallback to fixed-size chunking
        segments = chunk_text_fixed_size(raw_text, chunk_size=25000)
        boundary_decisions = None

    # 3. Create Records for each segment
    segment_records = []
    for i, seg in enumerate(segments):
        record = Record(
            type="webpage_segment",
            text=seg,
            metadata={
                'url': url,
                'segment_id': i,
                'boundary_decision': boundary_decisions[i] if boundary_decisions else None
            }
        )
        segment_records.append(record)

    return segment_records
```

### Reward for Dynamic Segmentation

```python
def compute_segmentation_quality_reward(episode_data):
    """
    Reward based on how well the segmentation helped the agent.
    """

    # 1. PRIMARY: Task success (as before)
    task_success = episode_data['task_success']  # 0.0 or 1.0

    # 2. AUXILIARY: Segment coherence
    segment_coherence = np.mean([
        compute_intra_chunk_coherence(seg)
        for seg in episode_data['segments']
    ])

    # 3. AUXILIARY: Information preservation
    #    Check if important information wasn't split across boundaries
    info_preservation = check_entity_preservation(episode_data['segments'])

    # 4. AUXILIARY: Efficiency (fewer, larger chunks vs. many tiny chunks)
    num_segments = len(episode_data['segments'])
    avg_segment_size = np.mean([len(seg) for seg in episode_data['segments']])
    efficiency = 1.0 / (1.0 + abs(num_segments - target_num_segments))

    # Combined reward
    reward = (
        task_success +                # Primary: 1.0 weight
        0.05 * segment_coherence +    # Auxiliary: 0.05
        0.1 * info_preservation +     # Auxiliary: 0.1
        0.05 * efficiency            # Auxiliary: 0.05
    )

    return reward
```

---

## Phase 3: End-to-End Hierarchical Memory (Weeks 5-6)

### Goal
Combine segmentation + selection into a unified hierarchical memory system.

```
Document → Dynamic Segmentation → Memory Pool → RL Selection → Context Window
```

### Architecture

```python
class HierarchicalMemorySystem:
    """
    Full memory system combining:
    1. Dynamic segmentation of incoming information
    2. Hierarchical organization of segments
    3. RL-based selection for context construction
    """

    def __init__(self):
        self.segmenter = DynamicMemorySegmenter()
        self.selector = MemorySegmentSelector()

        # Hierarchical memory structure
        self.memory_hierarchy = {
            'prompt': [],           # Level 0: Task description
            'reasoning': [],        # Level 1: Agent's reasoning steps
            'search_results': [],   # Level 2: Search result segments
            'webpages': []         # Level 3: Webpage segments
        }

    def add_webpage(self, url, text, question_context):
        """Add webpage with dynamic segmentation"""
        segments, decisions = self.segmenter.segment_text(text, question_context)

        for seg in segments:
            self.memory_hierarchy['webpages'].append({
                'url': url,
                'text': seg,
                'embedding': self.encode(seg),
                'timestamp': time.time()
            })

    def prepare_context(self, current_state, budget=24000):
        """Select segments across hierarchy to build context"""

        # 1. Always include prompt
        context_parts = [self.memory_hierarchy['prompt']]
        remaining_budget = budget - len(str(context_parts[0]))

        # 2. Select from other levels using RL policy
        all_segments = (
            self.memory_hierarchy['reasoning'] +
            self.memory_hierarchy['search_results'] +
            self.memory_hierarchy['webpages']
        )

        selected_indices, importance = self.selector(
            all_segments, current_state, remaining_budget
        )

        # 3. Organize selected segments by hierarchy
        for idx in selected_indices:
            context_parts.append(all_segments[idx])

        return context_parts, importance
```

---

## Implementation Roadmap

### Week 1: Infrastructure Setup
- [ ] Implement `MemorySegmentSelector` class
- [ ] Modify `AgentMemory.prepare_prompt()` to support RL selection
- [ ] Add segment importance tracking
- [ ] Implement segment-level reward computation

### Week 2: Training Pipeline
- [ ] Modify `arun_episode()` to use memory selector
- [ ] Implement policy gradient updates for selector
- [ ] Add logging for segment selection statistics
- [ ] Run baseline experiments (sliding window vs RL selection)

### Week 3: Dynamic Segmentation
- [ ] Implement `DynamicMemorySegmenter` class
- [ ] Integrate with webpage processing pipeline
- [ ] Test segmentation quality on sample webpages
- [ ] Implement boundary decision rewards

### Week 4: Joint Training
- [ ] Train segmenter and selector jointly
- [ ] Implement hierarchical reward structure
- [ ] Evaluate on GAIA validation set
- [ ] Analyze failure cases

### Week 5-6: Optimization & Evaluation
- [ ] Hyperparameter tuning
- [ ] Full evaluation on all benchmarks (GAIA, xBench, Frames)
- [ ] Ablation studies (segmentation only, selection only, both)
- [ ] Write up results

---

## Expected Challenges & Solutions

### Challenge 1: Sparse Rewards
**Problem**: Task success is binary (0/1), may not provide enough signal for segment-level learning.

**Solution**:
- Use intermediate rewards (F1 score instead of EM)
- Implement credit assignment to segments
- Use auxiliary coherence/efficiency rewards during early training

### Challenge 2: Long Training Time
**Problem**: Each episode is 128+ turns with 30k+ tokens—slow to collect trajectories.

**Solution**:
- Start with shorter episodes (force_turns=4, max_turns=32)
- Use smaller models for initial validation (Qwen-7B)
- Leverage ASearcher's async architecture for parallel rollouts

### Challenge 3: Reward Attribution
**Problem**: Hard to know which segments caused success/failure.

**Solution**:
- Track segment usage: which segments were actually attended to by LLM
- Use attention weights (if available) to assign credit
- Implement counterfactual evaluation (what if segment X was excluded?)

### Challenge 4: Computational Cost
**Problem**: Adding RL policy for segmentation increases compute.

**Solution**:
- Use lightweight encoders (MiniLM, not SONAR initially)
- Cache embeddings for repeated segments
- Train selector separately from main agent first

---

## Success Criteria

### Minimum Viable Success (Phase 1)
- [ ] RL selector achieves ≥ baseline task success rate
- [ ] Reduces average tokens used by ≥10%
- [ ] Shows interpretable segment selection patterns

### Target Success (Phase 2)
- [ ] Improves task success rate by +5-10% over baseline
- [ ] Reduces tokens by 20-30%
- [ ] Demonstrates multi-hop reasoning improvement

### Stretch Goals (Phase 3)
- [ ] Generalizes across different question types
- [ ] Transfers to other benchmarks (HotpotQA, Bamboogle)
- [ ] Enables longer episodes (256+ turns) without context overflow

---

## Key Files to Modify

1. **`/home/user/ASearcher/agent/asearcher.py`**
   - `AgentMemory` class: Add RL-based selection

2. **`/home/user/ASearcher/ASearcher/train/asearcher_reasoning.py`**
   - `arun_episode()`: Integrate memory selector
   - Training loop: Add selector updates

3. **`/home/user/ASearcher/ASearcher/utils/search_tool.py`**
   - Webpage processing: Add dynamic segmentation

4. **`/home/user/ASearcher/ASearcher/utils/rewards.py`**
   - Add segment-level reward functions

5. **New file: `/home/user/ASearcher/ASearcher/memory/segment_selector.py`**
   - Implement `MemorySegmentSelector` and `DynamicMemorySegmenter`

---

## Conclusion

ASearcher provides an **ideal validation platform** for your memory segmentation approach because:

1. ✅ **Clear task success signal**: Answer correctness is objective and measurable
2. ✅ **Existing RL infrastructure**: GRPO pipeline can be directly extended
3. ✅ **Real memory challenges**: 32k+ context with multi-hop reasoning
4. ✅ **Natural segmentation points**: Records provide structure
5. ✅ **Strong baselines**: Current sliding window approach to compare against

**Recommended Path**: Start with **Phase 1** (segment selection), as it:
- Requires minimal changes to existing code
- Provides quick validation of core idea
- Can be trained efficiently
- Gives clear signals about whether RL-based memory management helps

If Phase 1 succeeds, Phase 2 (dynamic segmentation) becomes a natural extension.

**This validation will directly test your key hypothesis**: Does optimizing memory segmentation for agent task success (via RL) improve performance compared to static heuristics?
