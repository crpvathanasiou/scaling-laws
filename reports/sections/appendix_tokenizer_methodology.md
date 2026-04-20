### Vocabulary Profiling Metrics

Let \( V \) denote the tokenizer vocabulary size.

**Fertility**
\[
\text{fertility} = \frac{\text{total number of subword tokens}}{\text{total number of whitespace-delimited words}}
\]

Lower fertility indicates less fragmentation.

**Embedding share**
For the smallest model with hidden size \( d \) and total trainable parameters \( N \), the approximate embedding share is:

\[
\text{embedding share} = \frac{V \cdot d}{N}
\]

when input and output embeddings are tied. If untied embeddings are used, the numerator becomes approximately \( 2 \cdot V \cdot d \).

**Active vocabulary ratio**
\[
\text{active vocab ratio} = \frac{\#\{t \in V : \text{count}(t) \ge k\}}{|V|}
\]

where \( k \) is a minimum usage threshold on the profiling subset.