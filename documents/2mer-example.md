# Conceptual understanding of basecalling
*The purpose of this document is to investigate the actual complexity of performing basecalling on data from an ONT MinION device through small but realistic examples.* 

From an outside perspective, the MinION can be seen as a tool for converting organic DNA/RNA material into electrical signals. Basecalling is the process of converting the electrical signals into a sequence of bases, i.e. a sequence over the letters `ACGT`.

The MinION device produces the electrical signal by pulling a piece of RNA through a scanner, called a *nanopore*, and *reading* five bases at a time. A *read* is the resistance measured while sending an electrical current through the five bases inside the nanopore. Basecalling is possible because the resistance varies based on which bases are currently being read.

Based on this description, basecalling might seem trivial, but there are certain complications, which increase the complexity of the problem. What follows is thus a number of small, tangible examples that will serve to uncover the complexities inherent in basecalling and allow for better discussions about the topic.


## Example 1: A Single Read
With an understanding of the big picture in place, we now delve into the details regarding how a *single* read from the nanopore is produced.

RNA is a sequence over the alphabet `Σ = {A, C, G, T}`. 

Example: 
> AACGTGTGACA

As described previously, a single read from the MinION device is the resistance measured when sending an electrical current through the *five* bases in the nanopore. In biology, a subsequence of length `k` is called a `k-mer`. A read is thus based on a `5-mer` of bases. Note that the reads are integer values.

Example: 
> AACGT => 354

This immediately brings a number of questions concerning the properties of a read to mind, which we will deal with next. 
1. Does the ordering of bases matter?
For example, is the following expected if the read is assumed to have *no noise*.

`AACGT => x        # Initial order`
`TGCAA => x        #  Reversed order`
`TAGAC => x        # Completely reordered`
Where x is an arbitrary integer.
2. Can partial bases be read?
For example:
`aGTCAg` where the lower-case `a` and `g` each represent a half base (therefore it still reads 5 bases in total).
    a. If partial bases can be read, is there some limit to how small a fraction of a base that can be read?
    b. And will the nanopore always hold five bases in total?
    For example: `1/3 + 1 + 1 + 1 + 1 + 2/3`, where a third of the first base is present and therefore the two-thirds of the last one is as well.
### Properties of the sequence
The encoded signal produced by the device is, therefore:
1. A sequence of integers defining the resistances of contexts
2. The signal for context $c_i$ is not affected by the signal of context $c_{i+y}$ where $y$ can be any given value.

In addition we will argue, that the following also holds:

3. The change between signal $s_i$ and $s_{i+1}$ provide *some information* about the change of the context:
    - The same of course goes for $s_{i+1}$ and $s_{i+2}$
    - **This means that:** The change between $s_i$ and $s_{i+2}$ also provide some information about the change in context
    - **But:** reads close to each other have higher correlation (as we will argue in our example)


## Questions
The signal output range from the MinION is somewhere between 200 and 850 yielding 650 possible values. DNA consists of 4 different bases of which may have 1 (or 2?) modifications, that affect their signal. Assuming that the nanopore always reads 5 (and only 5) bases, then this produces $5^5$ (or $5^6$) different combinations of contexts. In an ideal scenario, we would be able to make a function that takes a signal read and returns the correct context. This ideal scenario is not possible since it is impossible to create a function where the number of inputs (650) is lower than the number of outputs ($5^5$ or $5^6$).

This is just a testament to the complexity of the task of basecalling. To make valid arguments about a solution, we need a better understanding of the correlations between input and output of the MinION device.

- Does the data that we are provided include modified bases, or is that only done artificially?
- Does any combination of 5 bases produce the same read? - (or are bases closer to the middle more dominating?)
  - e.g. are these signals equal: ACGAA = AAACG
- If not, does the reverse of a 5-mer produce the same read?
  - e.g. are these signals equal: ACGAA = AAGCA
- Is there an instant switch between two contexts - or is it a continuous transition? See below:

  **IS THE FOLLOWING CORRECT?**
  In reality, the device will not measure a set of reads for a 5-mer and then the following read will be of a new 5-mer (e.i one base moved out and a new moved in). Instead, are the first couple of reads of a 5-mer also influenced by the 5-mer before it and the last reads are influenced by the 5-mer after.

  Example: 
  ```
  aACGTG  => 350
  aACGTG  => 350
  ACGTG  => 355
  ACGTG  => 355
  ACGTGt => 360
  ACGTGt => 360
  ```
  Where lowercase letters are only partial influencing the signal. Everything influencing a given read is refered to by the *context*, whereas a *5-mer* is the five capital letters.



## Example
Imagine the hypothetic DNA, that consists of the bases `A`, `B`, and `C` and a device that only captures 2-mers. Assume that the device is not influenced by noise and makes one read per 2-mer. These adjustments greatly simplify the problem, but every *property of the sequence* still holds.

Giving these assumptions we can make a table with the signal of every 2-mer (2^3 = 9 different possibilities).

| 2-mer  | AA | AB | AC | BA | BB | BC | CA | CB | CC |
|--------|----|----|----|----|----|----|----|----|----|
|**Read**| 1  | 1  | 1  | 3  | 2  | 3  | 4  | 4  | 2  |

If given the signal value 1, we can make a lookup in the table, and see that this either corresponds to AA, AB or AC. Only provided with a single read does not enable us to deterministically select the correct 2-mer - but we have narrowed it down. If the next read is also a 1, we know that the first 2-mer is AA, because the only way two reads of 1 can overlap, is if both 2-mers are AA. If the second read instead is 2, it means that first 2-mer is either AB or AC. We can use this process of overlapping k-mers until the correct initial one is determined (given that the values for the different k-mers are well distributed). The following shows the graph of how the sequence [1, 2, 4] deterministically chooses the 2-mer AC. 

<img src="@attachment/2mer.png">

It is obvious, that we cannot make an overlap if we skip over a 2-mer, since the two 2-mers will not share any bases. Extending the example to 3-mers (or higher) will show that skipping will still result in un-determinism. Take the example, that we are choosing whether the 3-mer is ABA or ACA. If we skip the read of the second 3-mer we can only make an overlap on the third base. Regardless of what 3-mer starting with A that the third read suggests, we would not be able to decide between ABA or ACA. Whereas an overlap from the second read would aid with that.

By this we can conclude when determining any k-mer $i$, that signal $S_{i+k}$ has more importance than signal $S_{i+k+1}$.
