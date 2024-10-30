# BGE Embeddings: Landmark Embeddings, Chunking-Free (18 Feb 2024) [Link]

## Highlights
- **Chunking-Free Architecture:** Produces high-quality embeddings over long contexts without chunking.
- **Position-Aware Objective Function:** Emphasizes boundaries for sequential spans of information.
- **Multi-Stage Learning Algorithm:** Details provided below.

### Why?
- The context window is limited, but the goal is to provide more context.
- Traditional RAG: Chunking → Embeddings → Retrieval.
- **Chunking is challenging:** Disconnected chunks break coherence, causing important chunks to be overlooked.
- **Solution:** Chunking-free method.

### What is What?

#### Chunking-Free
- Embeddings for the input are generated based on a coherent, long context.
- Introduces special tokens called Landmarks (LMKs), added at the end of each sentence.
- Utilizes the LLM encoder to process landmarked, long contexts.

#### Position-Aware Objective Function
- Groups of sentences often convey unified information.
- Instead of treating each sentence equally, the model assigns weights that grow exponentially based on the sentence's position in the context.
- Jointly selects `front-K` sentences.

#### Multi-Stage Learning Algorithm
- Two goals: **Semantic Discrimination** and **Contextualized Representation.**
- Achieved through **distant supervision** with pairwise data and **weak supervision** over noisy, long contexts synthesized by LLMs.

### Issues with Current Methods
- Current approaches involve modifying positional encoding mechanisms, enabling LLMs trained on shorter contexts to handle long inputs during inference.
- **Fine-tuning for longer contexts:** Costly and can affect short-sequence input handling.
- **Context compression and stream processing** can discard information outside the context window.

### Landmark Embeddings

#### Preliminary
- Typically, RAG extracts information, then sends chunks to the LLM for processing, breaking context coherence.

#### Chunking-Free Architecture
- LMK tokens are added at the end of each sentence.
- **See Figure 2** [Embed image here].
- Landmarks capture underlying semantics and are jointly encoded with sentences and neighboring contexts, producing Landmark Embeddings.
- Utilizes LLaMa-2-7B as the encoding backbone.

```
Landmark Embedding
LEi ←LLM(c1,...,ci; LMK).embed[−1], 
Query Embedding
Eq ←LLM(query; LMK).embed[−1].
```
- Relevnace is calculated as dot product of Eq ,LEi
- What if there is still a long context, obviously our backbone is LLM so there will be some context window associated with it. But we are conduct it as
- `LEi ←LLM(ci−l,...,ci; LMK).embed[−1],` here l is the number of sentences.

### Position Aware Objective
- The landmark embedding is learned by contrastive learning, where the query and its relevant sentences can be distinguished by the higher embedding similarities.
- Modified the contrastive function
- Introduced positional weight  wi ←exp(−α∗i), alpha is temprature param.
- Benefits? the relevant sentences can be fully utilized for the training of landmark embedding and the ultimate boundary of the useful information can be emphasized and better discriminated.

### Multi Stage Learning
- Typically, training includes question and answer pairs which turns out to be inappropriate for the training.
- They propose to factorised this with 2 fundamental capabilities
    - Basic semantic discriminability
    - Contextual represenatation
- Landmark embedding is initialized as a general sentence-level embedding model.
- Then it is enhanced as a contextual representation model where discriminative embeddings can be generated for its included sentences.

- Distant Supervision:
    - Only 1 landmark is appended in the sentence during training with 15 hard negative samples.
- Weak Supervision: 
    - With some modification in the pairwise training data, model can be trained to generate discriminative sentence embeddings within long context.
    - Randomly shuffle the answers from different queries and merge them as a pseudo long document. [Insert Image]
    - And use inbatch negatives.
- Fine tuning:
    - Leverage synthetic data
    - Series of text spans are randomly sampled from large document (Pulled from wikipedia) and pseudo query are generated from prompting LLM.
    - Small amount of data is generated for the training stage.

### Experiment
We have 3 questions
- The exploration of landmark embedding’s impact on the retrieval augmentation of long-context language modeling
- The comparison between landmark embedding and the existing retrieval methods based on chunked contexts.
- The analysis of technical factors in landmark embedding.
------------
- Compared LLAMA 2 7B 4k Context length and GPT-3.5 16k Context length.
- On many benchmarks this was the best by maintaing the same context length, look at the table.


### Main Results
#### Analysis on retrieval augmentation
- Outperformed the LLaMa 2-7B 
- Challenge: In GPT 3.5, the context window there is 16k and Retriever Context window is 2k
- Still outperformed GPT 3.5.
- By using 2 methods, one sliding window and other the LLM context. They observed sliding window works better for retrieval.
- According to the evaluation result, the position-aware objective with Front-k outperforms the ablation baselines in the downstream language modeling.
- For experimenatation they did an arbitrary combinations of different stages, They observed from the evaluation result, the third stage, i.e. the fine-tuning over synthetic data, presents the highest individual training effect. This result can probably be attributed to its closest relationship with the downstream task.
- Using all three gave them optimal empirical performance.


