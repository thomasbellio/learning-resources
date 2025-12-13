# Large Language Models: Conceptual Framework

## 1. Operative Definition of Large Language Models

**An LLM is:**
> "A system that learns statistical patterns from language data and uses those patterns to probabilistically generate language continuations."

### Key Aspects of This Definition:
- **Learns statistical patterns**: The model discovers regularities in language through training on large corpora
- **From language data**: The source material is text-based linguistic information
- **Probabilistically generate**: Output is not deterministic but based on probability distributions
- **Language continuations**: The next most probable language sequence given the statistical patterns learned and the current context

---

## 2. Core Definitions

### 2.1 Tokens

**Definition:** Discrete symbolic units (represented as integers) that serve as the atomic elements of text processing in an LLM.

**Key Properties:**
- Created through a tokenization algorithm (e.g., BPE, WordPiece, SentencePiece)
- Part of a fixed, pre-defined vocabulary
- Often represent subword units rather than complete words
- Serve as identifiers that map to vector embeddings

**Example:**
- Text: "unhappiness"
- Tokens: `["un", "happiness"]` or `["un", "happy", "ness"]`
- Token IDs: `[2341, 5892]` (integer representations)

### 2.2 Vector Embeddings

**Definition:** Continuous, dense numerical representations (vectors) that encode semantic and statistical properties of tokens or sequences of tokens.

**Key Properties:**
- High-dimensional (e.g., 768, 1024, 4096 dimensions)
- Learned during model training
- Capture semantic relationships in continuous space
- Enable mathematical operations on language

**Example:**
- Token "cat" (ID: 5421) → `[0.23, -0.47, 0.91, ..., 0.15]` (768-dimensional vector)

---

## 3. Tokens vs Vector Embeddings: Key Distinctions

| Aspect | Tokens | Vector Embeddings |
|--------|--------|-------------------|
| **Nature** | Discrete, symbolic (integers) | Continuous, numerical (real-valued vectors) |
| **Creation** | Pre-defined via tokenization algorithm | Learned during neural network training |
| **Purpose** | Symbolic identifiers for text units | Semantic/statistical representations |
| **Dimensionality** | Scalar (single integer) | High-dimensional vector (hundreds/thousands of dimensions) |
| **Mutability** | Fixed after vocabulary creation | Updated during training via backpropagation |
| **Space** | Discrete vocabulary space | Continuous vector space |

**Relationship:**
- Tokens are **mapped to** vector embeddings via an embedding matrix
- The embedding matrix is a learned component of the neural network
- Vector embeddings serve as the bridge between discrete symbolic tokens and continuous neural network processing

---

## 4. Revised Conceptual Model of LLM Operation

### Overview
The LLM operation can be understood as a five-step process that transforms user input text into generated output text through tokenization, neural network processing, and detokenization.

---

### 4.1 User Input
**Description:** The process begins when a user provides natural language text as input to the system.

**Characteristics:**
- Raw, human-readable text
- Can be a question, instruction, prompt, or partial text
- Represents the context for generation

---

### 4.2 Tokenization (Text → Tokens)
**Description:** A separate tokenizer component (not part of the neural network) converts the input text into a sequence of token IDs using deterministic lookup.

**Process:**
1. Text is broken into subword units based on a pre-defined vocabulary
2. Each subword is mapped to its corresponding integer token ID
3. The result is a sequence of integers

**Key Points:**
- Completely deterministic (same input always produces same tokens)
- Uses algorithms like BPE, WordPiece, or SentencePiece
- No learning involved—pure lookup and rule application
- The vocabulary is fixed and created before model training

**Example:**
- Input: "The cat sat"
- Output: `[5421, 8934, 2847]`

---

### 4.3 Neural Network Processing (Token Prediction)
**Description:** The model leverages the token sequence and learned statistical patterns to autoregressively predict the next most likely token, adding it to the context and repeating.

**Detailed Process:**

#### 4.3.1 Embedding Lookup
- Each token ID is mapped to its corresponding vector embedding
- This uses a learned embedding matrix (part of the neural network)
- Token `5421` → `[0.23, -0.47, 0.91, ..., 0.15]`

#### 4.3.2 Transformer Processing
- Sequence of embeddings flows through multiple neural network layers
- **Self-attention mechanisms**: Tokens contextualize each other
- **Feed-forward networks**: Non-linear transformations
- **Layer normalization**: Stabilization
- Each layer builds increasingly abstract representations

#### 4.3.3 Output Prediction
- Final layer produces a probability distribution over the entire vocabulary
- Each token in the vocabulary receives a probability score
- Scores sum to 1.0 across all possible tokens

#### 4.3.4 Autoregressive Generation
- Sample or select the next token based on the probability distribution
- Add the predicted token to the sequence
- Repeat the process with the extended sequence
- Continue until a stopping condition (end token, max length, etc.)

**Key Points:**
- The model only works with token IDs and their embeddings—never raw text
- Generation is sequential, one token at a time
- Each prediction is conditioned on all previous tokens
- Learned parameters (weights) encode the statistical patterns from training

---

### 4.4 Detokenization (Tokens → Text)
**Description:** The tokenizer converts the predicted sequence of token IDs back into human-readable text through reverse lookup.

**Process:**
1. Each token ID is mapped back to its corresponding string
2. Strings are concatenated with appropriate spacing
3. The result is natural language text

**Key Points:**
- Deterministic reverse lookup process
- No learning involved
- Mirrors the tokenization process in reverse

**Example:**
- Input: `[8934, 5421, 9823]`
- Output: "sat on the"

---

### 4.5 Output to User
**Description:** The generated text is returned to the user as the model's response.

**Characteristics:**
- Human-readable natural language
- Represents a probabilistic continuation of the input context
- Generated autoregressively based on learned statistical patterns

---

## 5. System Architecture Summary

```
User Input (Text)
       ↓
[Tokenizer] ← Pre-defined vocabulary
       ↓
Token Sequence (Integers)
       ↓
[Embedding Matrix] ← Learned parameters
       ↓
Vector Embeddings
       ↓
[Neural Network Layers] ← Learned parameters
  - Self-attention
  - Feed-forward
  - Layer normalization
       ↓
Probability Distribution over Vocabulary
       ↓
[Sampling/Selection]
       ↓
Predicted Token Sequence
       ↓
[Tokenizer - Reverse Lookup]
       ↓
Output Text
       ↓
User Output (Text)
```

---

## 6. Key Takeaways

1. **LLMs are statistical pattern learners** that generate language probabilistically based on learned patterns
2. **Tokenization is separate from the model** - it's a deterministic preprocessing/postprocessing step
3. **Tokens are symbolic identifiers**, while **embeddings are semantic representations**
4. **The neural network never sees raw text** - it only processes token IDs and their embeddings
5. **Generation is autoregressive** - one token at a time, with each prediction conditioned on previous tokens
6. **The vocabulary is fixed** and defines the complete space of atomic units the LLM can process

---

*Document created from conceptual discussion on LLM architecture and operation*