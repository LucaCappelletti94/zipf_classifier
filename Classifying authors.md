
# Zipf Classifier
## Using Zipfs to classify books and articles

## Introduction
Hello, I'm Luca Cappelletti and here I will show you a complete explanation and example of usage of [ZipfClassifier](https://github.com/LucaCappelletti94/zipf_classifier), a classifier that leverages the assumption that some kind of datasets (texts, [some images](http://www.dcs.warwick.ac.uk/bmvc2007/proceedings/CD-ROM/papers/paper-288.pdf), even [sounds in spoken languages](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033993)) follows the [Zipf law](https://en.wikipedia.org/wiki/Zipf%27s_law).

## How to use this notebook

This is a [Jupyter Notebook](http://jupyter.org/). You can either read it [here on github](https://github.com/LucaCappelletti94/zipf_classifier/blob/master/Classifying%20authors.ipynb) or, **preferably** to enjoy all its aspects, run it on your own computer. Jupyter comes installed with [Anaconda](https://anaconda.org/), to execute it you just need to run the following in your terminal:

`jupyter-notebook`

## What we will use

### The packages
We will use obviously the [ZipfClassifier](https://github.com/LucaCappelletti94/zipf_classifier) and other two packages of mine: [Zipf](https://github.com/LucaCappelletti94/zipf) to create the distributions from the texts and [Dictances](https://github.com/LucaCappelletti94/dictances) for the classifications metrics. If you need to install them just run the following command in your terminal:

```pip install zipf dictances zipf_classifier```


```python
from zipf.factories import ZipfFromDir
from zipf_classifier import ZipfClassifier
from dictances import jensen_shannon, normal_total_variation, intersection_total_variation, kullback_leibler, intersection_squared_hellinger, hellinger, squared_hellinger

```

### Additional packages
We will also be using some utilities, such as the loading bar `tqdm` and the `requests` package. If you don't have them already you can install them by running:

```
pip install tqdm requests tabulate
```

The others packages should be already installed with python by default.


```python
import io
import inspect
import json
import math
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pprint import pprint

import requests
from tqdm import tqdm_notebook as tqdm

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from IPython.display import HTML, display
import tabulate

def tqdm(it, unit=''):
    return it
```

### Some small helpers
Let's make ome small functions to help out loading folders:


```python
def get_dirs(root):
    """Return subdirectories under a directory."""
    return [root+"/"+d for d in os.listdir(root) if os.path.isdir(root+"/"+d)]
```

and the book folders:


```python
def get_books(root):
    """Return all books found under a given root."""
    return [book[0] for book in os.walk(root) for chapter in book[2][:1] if chapter.endswith('.txt')]
```

and the saved zipfs:


```python
def get_zipfs(root):
    """Return all zipfs found under a given root."""
    return [zipfs[0]+"/"+zipf for zipfs in os.walk(root) for zipf in zipfs[2] if zipf.endswith('.json')]
```

### Some stylers


```python
frame_number = 30
```


```python
def b(string):
    """Return a boldified string."""
    return "\033[1m%s\033[0;0m"%string
```


```python
def red(string):
    """Return a red string."""
    return "\033[0;31m%s\033[0;0m"%string
```


```python
def yellow(string):
    """Return a yellow string."""
    return "\033[0;33m%s\033[0;0m"%string
```


```python
def green(string):
    """Return a green string."""
    return "\033[0;32m%s\033[0;0m"%string
```


```python
def gray(string):
    """Return a gray string."""
    return "\033[0;37m%s\033[0;0m"%string
```


```python
def print_function(function):
    """Print the source of a given function."""
    code = inspect.getsource(function)
    formatter = HtmlFormatter()
    display(HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter))))
```


```python
def success(results, metric):
    """Show the result of a given test."""
    successes = results["success"]
    total = successes + results["failures"] + results["unclassified"]
    percentage = round(successes/total*100,2)
    if percentage > 85:
        metric_name = green(metric.__name__)
    elif percentage > 70:
        metric_name = yellow(metric.__name__)
    else:
        metric_name = red(metric.__name__)
    print("Success with metric %s: %s"%(metric_name,b(str(percentage)+"%")))
    display(HTML(tabulate.tabulate(list(results.items()), ["Info", "Values"], tablefmt='html')))
```

### The datasets
I've prepared three **datasets**:

#### Authors dataset
Dataset of english books from three **famous authors**: **D. H. Lawrence**, **Oscar Wilde** and **Mark Twain**.

This dataset will be used to build a classifier able to classify the books to the respective author.

#### Periods dataset
Dataset of english books from four **style periods**: **Modernism**, **Naturalism**, **Realism** and **Romanticism**. 
This dataset will be used to build a classifier able to classify the books to the respective style period.

#### Recipes dataset
Dataset of italian articles, some containing recipes and some containing food reviews, food descriptions (eg wikipedia) or other articles.

We will use this to classify articles to **recipes** and **non recipes**.

### Retrieving the datasets
We download and extract the datasets:

- [Link to authors dataset](https://mega.nz/#!SS4jgDKJ!zoBJ-sP22_qfI5YUquTewzeZ2KsuLqSvM1u_UL--46A)
- [Link to periods dataset](https://mega.nz/#!2WREgZpQ!nDItooUUwVyGFyDBw1TA5TJ3GZGR7dzmJ4LbALvsiNs)
- [Link to recipes dataset](https://mega.nz/#!vLZUwTLB!LkK_ZdA8D3loowd3byw7CisZrDPkbcOPjBq1lu2PbnA)

Put them in the same folder of this notebook to use the datasets.


```python
datasets = ["authors", "periods", "recipes"]
```

Before going any further, let's check if the dataset are now present:


```python
for dataset in datasets:
    if not os.path.isdir(dataset):
        raise FileNotFoundError("The dataset %s is missing!"%(red(dataset)))
```

Ok! We can proceed.

#### Splitting into train and test
Let's say we leave 60% to learning and 40% to testing. Let's proceed to split the dataset in two:


```python
learning_percentage = 0.6
```

First we check if the dataset is already split (this might be a re-run):


```python
def is_already_split(root):
    """Return a bool indicating if the dataset has already been split."""
    split_warns = ["learning", "testing"]
    for sub_dir in os.listdir(root):
        for split_warn in split_warns:
            if split_warn in sub_dir:
                return True
    return False
```

Then we split the dataset's books as follows:

Since we want the zipfs that the classifier will use to do the classification built on a proportioned dataset, we pick the percentage of books put aside for learning from the minimum number of books for class in the dataset.


```python
def split_books(root, percentage):
    """Split the dataset into learning and testing."""
    min_books = math.inf
    for book_class in get_dirs(root):
        books = get_books(book_class)
        min_books = min(min_books, len(books))
    for book_class in get_dirs(root):
        books = get_books(book_class)
        random.seed(42) # for reproducibility
        random.shuffle(books) # Shuffling books
        n = int(min_books*percentage)
        learning_set, testing_set = books[:n], books[n:] # splitting books into the two partitions
        # Moving into respective folders
        [shutil.copytree(book, "%s/learning/%s"%(root, book[len(root)+1:])) for book in learning_set]
        [shutil.copytree(book, "%s/testing/%s"%(root, book[len(root)+1:])) for book in testing_set]
```

Here we actually run the two functions:


```python
for dataset in datasets:
    if is_already_split(dataset):
        print("I believe I've already split the dataset %s!"%(b(dataset)))
    else:
        split_books(dataset, learning_percentage)
```

    I believe I've already split the dataset [1mauthors[0;0m!
    I believe I've already split the dataset [1mperiods[0;0m!
    I believe I've already split the dataset [1mrecipes[0;0m!


## The metrics

We will use metrics for distributions $P, Q$ that hold the following properties:

$$
    q_i > 0\quad \forall i \in Q, \qquad p_i > 0 \quad \forall i \in P \qquad \sum_{i \in Q} q_i = 1, \qquad \sum_{i \in P} p_i = 1
$$

These metrics must wither have computational complexity $O(\min{(n,m)}))$ (where $n$ and $m$ are respectively the cardinality of distributions $P$ and $Q$) or be defined only on the intersection of the distributions, for being practically usable.

Informations on the metrics used are below:

### Kullback Leibler Divergence
$$
    D_{KL}(P,Q) = \sum_i P(i) \log{\frac{P(i)}{Q(i)}}
$$
The KL divergece is defined for all events in a set $P, Q \subseteq X$. 

This forces to define the KL for zipfs only on the subset of the events that are shared beetween the two distributions: $X = P \cap Q$.

This ignores all the information about non-sharec events and it is solved via the Jensen Shannon divergence.

## Jensen Shannon Divergence
$$
    JSD(P,Q) = \frac{1}{2}D(P,M) + \frac{1}{2}D(Q, M) \qquad M = \frac{1}{2}(P+Q)
$$

The KL divergence is defined for every event in a set $X=P\cup Q$, it is **symmetric** and has **always a finite value**.

### Getting the current implementation

The current implementation works as follows:

Starting from the extended formulation:

$$
    m_i = \frac{1}{2}(p_i+q_i), \quad p_i = \begin{cases}
        p_i & i \in P\\
        0 & otherwise
    \end{cases}, \quad q_i = \begin{cases}
        q_i & i \in Q\\
        0 & otherwise
    \end{cases}
$$

$$
    JSD(P,Q) = \frac{1}{2}\sum_{i \in P} p_i \log{\frac{p_i}{m_i}} + \frac{1}{2}\sum_{j \in Q} q_j \log{\frac{q_j}{m_j}}
$$

Replacing in the formulation $m_i$:

$$
    JSD(P,Q) = \frac{1}{2}\sum_{i \in P} p_i \log{\frac{p_i}{\frac{1}{2}(p_i+q_i)}} + \frac{1}{2}\sum_{j \in Q} q_j \log{\frac{q_j}{\frac{1}{2}(p_j+q_j)}}
$$

Splitting the sums in 3 parts: $i \in P\setminus P\cap Q$, $i \in P\cap Q$ and $i \in Q\setminus P\cap Q$.

$$
    JSD(P,Q) = JSD_1(P,Q) + JSD_2(P,Q) + JSD_3(P,Q)
$$

\begin{align}
    JSD_1(P,Q) &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{\frac{p_i}{\frac{1}{2}(p_i+q_i)}} + \frac{1}{2}\sum_{j \in P\setminus P\cap Q} q_j \log{\frac{q_j}{\frac{1}{2}(p_j+q_j)}}\\
               &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{\frac{p_i}{\frac{1}{2}(p_i+q_i)}}\\
               &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{\frac{p_i}{\frac{1}{2}p_i}}\\
               &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{\frac{1}{\frac{1}{2}}}\\
               &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{2}\\
               &= \frac{1}{2}\log{2}\sum_{i \in P\setminus P\cap Q} p_i\\
\end{align}

\begin{align}
    JSD_2(P,Q) &= \frac{1}{2}\sum_{i \in P\setminus P\cap Q} p_i \log{\frac{p_i}{\frac{1}{2}(p_i+q_i)}} + \frac{1}{2}\sum_{j \in P\setminus P\cap Q} q_j \log{\frac{q_j}{\frac{1}{2}(p_j+q_j)}}\\
               &= \frac{1}{2}\sum_{i \in P\cap Q} p_i \log{\frac{p_i}{\frac{1}{2}(p_i+q_i)}} + q_i \log{\frac{q_i}{\frac{1}{2}(p_i+q_i)}}\\
               &= \frac{1}{2}\sum_{i \in P\cap Q} p_i \log{\frac{2p_i}{p_i+q_i}} + q_i \log{\frac{2q_i}{p_i+q_i}}\\
\end{align}

\begin{align}
    JSD_3(P,Q) &= \frac{1}{2}\log{2}\sum_{j \in Q\setminus P\cap Q} q_j\\
\end{align}

Summing $JSD_1$ and $JSD_3$ we can obtain:

$JSD_1+JSD_3 = \frac{1}{2}\log{2}\left(\sum_{i \in P\setminus P\cap Q} p_i + \sum_{j \in Q\setminus P\cap Q} q_j\right)$

In particular, if $\sum_{j\in Q}^m q_j = 1$ and $\sum_{i\in P}^n p_i = 1$, we can write:

\begin{align}
JSD_1+JSD_3 &= \frac{1}{2}\log{2}\left(2 - \sum_{i \in P\cap Q} p_i - \sum_{j \in P\cap Q} q_j\right)\\
            &= \frac{1}{2}\log{2}\left(2 - \sum_{i \in P\cap Q} p_i + q_j\right)\\
\end{align}

Putting all togheter we obtain:

$JSD(P,Q) =  \frac{1}{2}\left[\sum_{i \in P\cap Q} \left(p_i \log{\frac{2p_i}{p_i+q_i}} + q_i \log{\frac{2q_i}{p_i+q_i}}\right) + \log{2}\left(2 - \sum_{i \in P\cap Q} p_i + q_j\right)\right]$

What's marvelous about this semplification is that the computational complexity decrease from a naive literal interpretation of the initial formula of $O(n+m)$ to $O(\min(n,m))$ simply choosing to iterate over whichever of the two distributions holds less events.

The process is nearly identical for all other metrics shown below:

## Hellinger

Given two distributions $P, Q \subseteq X$, the **Hellinger distance** is defined as follows:

\begin{align}
    H(P,Q) = \frac{1}{\sqrt{2}}\sqrt{\sum_{i \in X} \left(\sqrt{p_i} - \sqrt{q_i} \right)^2}
\end{align}

### Achieving the current implementation
Given $\sum_{i\in P}^n p_i = 1$ and $\sum_{i\in Q}^m q_i = 1$, we can proceed by separating the sum inside the squared root into three partitions of $X$: $i \in P\setminus P\cap Q$, $i \in P\cap Q$ and $i \in Q\setminus P\cap Q$.

\begin{align}
    H(P,Q) = \frac{1}{\sqrt{2}}\sqrt{H_1(P,Q) + H_2(P,Q) + H_3(P,Q)}
\end{align}

Recalling the definitions of $p_i, q_i$:

$$p_i = \begin{cases}
        p_i & i \in P\\
        0 & otherwise
    \end{cases}, \quad q_i = \begin{cases}
        q_i & i \in Q\\
        0 & otherwise
    \end{cases}$$

We begin from $H_1(P,Q)$, for $i \in P\setminus P\cap Q$:
\begin{align}
    H_1(P,Q) &= \sum_{i \in P\setminus P\cap Q} \left(\sqrt{p_i} - \sqrt{q_i} \right)^2\\
             &= \sum_{i \in P\setminus P\cap Q} \left(\sqrt{p_i} \right)^2\\
             &= \sum_{i \in P\setminus P\cap Q} p_i\\
             &= 1 - \sum_{i \in P\cap Q} p_i
\end{align}

We solve $H_2(P,Q)$, for $i \in P\cap Q$:
$$
\sum_{i \in P\cap Q} \left(\sqrt{p_i} - \sqrt{q_i} \right)^2
$$

Now we solve $H_3(P,Q)$, for $i \in Q\setminus P\cap Q$:

\begin{align}
    H_3(P,Q) &= \sum_{i \in Q\setminus P\cap Q} q_i\\
             &= 1 - \sum_{i \in P\cap Q} q_i
\end{align}

Now, putting it all togheter we have:

\begin{align}
    H_1(P,Q) + H_2(P,Q) + H_3(P,Q) &=  2 + \sum_{i \in P\cap Q} \left[\left(\sqrt{p_i} - \sqrt{q_i} \right)^2 -p_i - q_i \right]\\
        &= 2 + \sum_{i \in P\cap Q} -2\sqrt{p_i q_i}\\
        &= 2 \left(1 - \sum_{i \in P \cap Q} \sqrt{p_i q_i} \right)
\end{align}

So the Hellinger distance is redefined as:

\begin{align}
    H(P,Q) &= \frac{1}{\sqrt{2}}\sqrt{2 \left(1 - \sum_{i \in P \cap Q} \sqrt{p_i q_i} \right)}\\
           &= \sqrt{1 - \sum_{i \in P \cap Q} \sqrt{p_i q_i}}
\end{align}


```python
metrics = [
    jensen_shannon,
    normal_total_variation,
    intersection_total_variation,
    kullback_leibler,
    intersection_squared_hellinger,
    hellinger,
    squared_hellinger
]
```


```python
for metric in metrics:
    print_function(metric)
```


<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">jensen_shannon</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span><span class="o">-&gt;</span><span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Return the jensen shannon divergence beetween a and b.&quot;&quot;&quot;</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">delta</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">big</span><span class="p">,</span> <span class="n">small</span> <span class="o">=</span> <span class="n">sort</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>

    <span class="n">big_get</span> <span class="o">=</span> <span class="n">big</span><span class="o">.</span><span class="fm">__getitem__</span>

    <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">small_value</span> <span class="ow">in</span> <span class="n">small</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">big_value</span> <span class="o">=</span> <span class="n">big_get</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">big_value</span><span class="p">:</span>
                <span class="n">denominator</span> <span class="o">=</span> <span class="p">(</span><span class="n">big_value</span> <span class="o">+</span> <span class="n">small_value</span><span class="p">)</span> <span class="o">/</span> <span class="mi">2</span>
                <span class="n">total</span> <span class="o">+=</span> <span class="n">small_value</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="n">small_value</span> <span class="o">/</span> <span class="n">denominator</span><span class="p">)</span> <span class="o">+</span> \
                    <span class="n">big_value</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="n">big_value</span> <span class="o">/</span> <span class="n">denominator</span><span class="p">)</span>
                <span class="n">delta</span> <span class="o">+=</span> <span class="n">big_value</span> <span class="o">+</span> <span class="n">small_value</span>
        <span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
            <span class="k">pass</span>

    <span class="n">total</span> <span class="o">+=</span> <span class="p">(</span><span class="mi">2</span> <span class="o">-</span> <span class="n">delta</span><span class="p">)</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">total</span> <span class="o">/</span> <span class="mi">2</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">normal_total_variation</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Determine the Normalized Total Variation distance.&quot;&quot;&quot;</span>
    <span class="n">big</span><span class="p">,</span> <span class="n">small</span> <span class="o">=</span> <span class="n">sort</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="n">big_get</span> <span class="o">=</span> <span class="n">big</span><span class="o">.</span><span class="fm">__getitem__</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">2</span>
    <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">small_value</span> <span class="ow">in</span> <span class="n">small</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">big_value</span> <span class="o">=</span> <span class="n">big_get</span><span class="p">(</span><span class="n">k</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">big_value</span><span class="p">:</span>
                <span class="n">total</span> <span class="o">+=</span> <span class="nb">abs</span><span class="p">(</span><span class="n">big_value</span> <span class="o">-</span> <span class="n">small_value</span><span class="p">)</span> <span class="o">-</span> <span class="n">big_value</span> <span class="o">-</span> <span class="n">small_value</span>
        <span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
            <span class="k">pass</span>
    <span class="k">return</span> <span class="n">total</span> <span class="o">/</span> <span class="mi">2</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">intersection_total_variation</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">overlap</span><span class="p">:</span> <span class="nb">bool</span><span class="o">=</span><span class="bp">False</span><span class="p">)</span><span class="o">-&gt;</span><span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Return the total distance beetween the intersection of a and b.&quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="n">intersection_nth_variation</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">overlap</span><span class="p">)</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">kullback_leibler</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Determine the Kullback Leibler divergence.&quot;&quot;&quot;</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">big</span><span class="p">,</span> <span class="n">small</span> <span class="o">=</span> <span class="n">sort</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="n">big_get</span> <span class="o">=</span> <span class="n">big</span><span class="o">.</span><span class="fm">__getitem__</span>
    <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">small_value</span> <span class="ow">in</span> <span class="n">a</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">big_value</span> <span class="o">=</span> <span class="n">big_get</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">big_value</span><span class="p">:</span>
                <span class="n">total</span> <span class="o">+=</span> <span class="n">small_value</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="n">small_value</span> <span class="o">/</span> <span class="n">big_value</span><span class="p">)</span>
        <span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
            <span class="k">pass</span>

    <span class="k">return</span> <span class="n">total</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">intersection_squared_hellinger</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Determine the Intersection Squared Hellinger distance.&quot;&quot;&quot;</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">big</span><span class="p">,</span> <span class="n">small</span> <span class="o">=</span> <span class="n">sort</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="n">big_get</span> <span class="o">=</span> <span class="n">big</span><span class="o">.</span><span class="fm">__getitem__</span>
    <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">small_value</span> <span class="ow">in</span> <span class="n">small</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">total</span> <span class="o">+=</span> <span class="p">(</span><span class="n">small_value</span><span class="o">**</span><span class="p">(</span><span class="mi">1</span> <span class="o">/</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-</span> <span class="n">big_get</span><span class="p">(</span><span class="n">key</span><span class="p">)</span><span class="o">**</span><span class="p">(</span><span class="mi">1</span> <span class="o">/</span> <span class="mi">5</span><span class="p">))</span><span class="o">**</span><span class="mi">2</span>
        <span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
            <span class="k">pass</span>
    <span class="k">return</span> <span class="n">total</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">hellinger</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Determine the Hellinger distance.&quot;&quot;&quot;</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="n">v</span> <span class="o">=</span> <span class="n">squared_hellinger</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">sqrt</span><span class="p">(</span><span class="n">v</span><span class="p">)</span>
    <span class="k">except</span> <span class="ne">ValueError</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">isclose</span><span class="p">(</span><span class="n">v</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">abs_tol</span><span class="o">=</span><span class="mf">1e-15</span><span class="p">):</span>
            <span class="k">return</span> <span class="mi">0</span>
        <span class="k">raise</span> <span class="n">e</span>
</pre></div>




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">squared_hellinger</span><span class="p">(</span><span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="sd">&quot;&quot;&quot;Determine the Squared Hellinger distance.&quot;&quot;&quot;</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">1</span>
    <span class="n">big</span><span class="p">,</span> <span class="n">small</span> <span class="o">=</span> <span class="n">sort</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="n">big_get</span> <span class="o">=</span> <span class="n">big</span><span class="o">.</span><span class="fm">__getitem__</span>
    <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">small_value</span> <span class="ow">in</span> <span class="n">small</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">total</span> <span class="o">-=</span> <span class="n">sqrt</span><span class="p">(</span><span class="n">small_value</span> <span class="o">*</span> <span class="n">big_get</span><span class="p">(</span><span class="n">key</span><span class="p">))</span>
        <span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
            <span class="k">pass</span>
    <span class="k">return</span> <span class="n">total</span>
</pre></div>



## The options
We will use the following options for learning and testing. More informations about options' customizations is available [here](https://github.com/LucaCappelletti94/zipf).


```python
options = {}
```

## Creating the Zipfs
We will now convert all the chapters in the dataset into the respective zipf for each option.


```python
def create_zipfs(paths, factory, test_path):
    for data_path in tqdm(paths, unit=' zipf'):
        path = "%s/%s.json"%(test_path, '/'.join(data_path.split('/')[1:]))
        # If the zipf already exists we skip it
        if os.path.exists(path):
            continue
        path_dirs = '/'.join(path.split('/')[:-1])
        zipf = factory.run(data_path, ['txt'])
        if not zipf.is_empty():
            if not os.path.exists(path_dirs):
                os.makedirs(path_dirs)
            zipf.save(path)
```

We define the paths for zipfs and their sources:


```python
def get_build_paths(dataset):
    """Return a triple with the build paths for given dataset."""
    learning_path = "%s/learning"%dataset
    testing_path = "%s/testing"%dataset
    zipfs_path = '%s/zipfs'%dataset

    print("I will build learning zipfs from %s,\ntesting zipfs from %s\nand save them in %s\n"%(b(learning_path), b(testing_path), b(zipfs_path)))
    return learning_path, testing_path, zipfs_path
```

First we create the learning zipfs:


```python
def build_learning_zipfs(path, zipfs_path):
    """Build zipfs from txt files at given path."""
    print("Creating learning zipfs in %s"%(b(path)))
    book_classes = get_dirs(path)
    print("Some of the paths I'm converting are:")
    random.seed(42) # For reproducibility
    random.shuffle(book_classes)
    shown = []
    for book in book_classes[:10]:
        shown.append((book, ''))
    display(HTML(tabulate.tabulate(shown, ["Learning data paths"], tablefmt='html')))
    create_zipfs(book_classes, factory, zipfs_path)
```

And then the testing zipfs:


```python
def build_testing_zipfs(path, zipfs_path):
    """Build zipfs from txt files at given path."""
    print("Creating testing zipfs in %s"%(b(path)))
    books = get_books(path)
    random.seed(42) # For reproducibility
    random.shuffle(books)
    shown = []
    for book in books[:10]:
        shown.append((book, ''))
    display(HTML(tabulate.tabulate(shown, ["Testing data paths", ''], tablefmt='html')))
    create_zipfs(books, factory, zipfs_path)
```

We create a factory for creating the zipfs objects from files with the options defined above. More informations about factory customization and other possible factories is available [here](https://github.com/LucaCappelletti94/zipf).


```python
factory = ZipfFromDir(options=options)
print("Created a factory with options %s"%(factory))
```

    Created a factory with options {
      "remove_stop_words": false,
      "stop_words": "it",
      "minimum_count": 0,
      "chain_min_len": 1,
      "chain_max_len": 1,
      "chaining_character": " ",
      "sort": false
    }


Wake up zipfs factory daemons:


```python
factory.start_processes()
```

Actually creating the zipfs:


```python
for dataset in datasets:
    print("Building dataset %s"%(b(dataset)))
    learning_path, testing_path, zipfs_path = get_build_paths(dataset)
    build_learning_zipfs(learning_path, zipfs_path)
    build_testing_zipfs(testing_path, zipfs_path)
    print(gray('='*frame_number))
```

    Building dataset [1mauthors[0;0m
    I will build learning zipfs from [1mauthors/learning[0;0m,
    testing zipfs from [1mauthors/testing[0;0m
    and save them in [1mauthors/zipfs[0;0m
    
    Creating learning zipfs in [1mauthors/learning[0;0m
    Some of the paths I'm converting are:



<table>
<thead>
<tr><th>                            </th><th>Learning data paths  </th></tr>
</thead>
<tbody>
<tr><td>authors/learning/twain      </td><td>                     </td></tr>
<tr><td>authors/learning/dh_lawrence</td><td>                     </td></tr>
<tr><td>authors/learning/wilde      </td><td>                     </td></tr>
</tbody>
</table>


    Creating testing zipfs in [1mauthors/testing[0;0m



<table>
<thead>
<tr><th>Testing data paths                                 </th><th>  </th></tr>
</thead>
<tbody>
<tr><td>authors/testing/twain/3275                         </td><td>  </td></tr>
<tr><td>authors/testing/twain/320                          </td><td>  </td></tr>
<tr><td>authors/testing/wilde/florentine-tragedy           </td><td>  </td></tr>
<tr><td>authors/testing/dh_lawrence/4483                   </td><td>  </td></tr>
<tr><td>authors/testing/wilde/2252                         </td><td>  </td></tr>
<tr><td>authors/testing/twain/3297                         </td><td>  </td></tr>
<tr><td>authors/testing/wilde/2305                         </td><td>  </td></tr>
<tr><td>authors/testing/dh_lawrence/fantasia-of-unconscious</td><td>  </td></tr>
<tr><td>authors/testing/wilde/2317                         </td><td>  </td></tr>
<tr><td>authors/testing/twain/3259                         </td><td>  </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m
    Building dataset [1mperiods[0;0m
    I will build learning zipfs from [1mperiods/learning[0;0m,
    testing zipfs from [1mperiods/testing[0;0m
    and save them in [1mperiods/zipfs[0;0m
    
    Creating learning zipfs in [1mperiods/learning[0;0m
    Some of the paths I'm converting are:



<table>
<thead>
<tr><th>                            </th><th>Learning data paths  </th></tr>
</thead>
<tbody>
<tr><td>periods/learning/romanticism</td><td>                     </td></tr>
<tr><td>periods/learning/realism    </td><td>                     </td></tr>
<tr><td>periods/learning/naturalism </td><td>                     </td></tr>
<tr><td>periods/learning/modernism  </td><td>                     </td></tr>
</tbody>
</table>


    Creating testing zipfs in [1mperiods/testing[0;0m



<table>
<thead>
<tr><th>Testing data paths                     </th><th>  </th></tr>
</thead>
<tbody>
<tr><td>periods/testing/romanticism/579        </td><td>  </td></tr>
<tr><td>periods/testing/modernism/3452         </td><td>  </td></tr>
<tr><td>periods/testing/romanticism/550        </td><td>  </td></tr>
<tr><td>periods/testing/romanticism/2124       </td><td>  </td></tr>
<tr><td>periods/testing/romanticism/4545       </td><td>  </td></tr>
<tr><td>periods/testing/romanticism/491        </td><td>  </td></tr>
<tr><td>periods/testing/modernism/3484         </td><td>  </td></tr>
<tr><td>periods/testing/romanticism/143        </td><td>  </td></tr>
<tr><td>periods/testing/modernism/blanco-posnet</td><td>  </td></tr>
<tr><td>periods/testing/realism/indian-summer  </td><td>  </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m
    Building dataset [1mrecipes[0;0m
    I will build learning zipfs from [1mrecipes/learning[0;0m,
    testing zipfs from [1mrecipes/testing[0;0m
    and save them in [1mrecipes/zipfs[0;0m
    
    Creating learning zipfs in [1mrecipes/learning[0;0m
    Some of the paths I'm converting are:



<table>
<thead>
<tr><th>                            </th><th>Learning data paths  </th></tr>
</thead>
<tbody>
<tr><td>recipes/learning/recipes    </td><td>                     </td></tr>
<tr><td>recipes/learning/non_recipes</td><td>                     </td></tr>
</tbody>
</table>


    Creating testing zipfs in [1mrecipes/testing[0;0m



<table>
<thead>
<tr><th>Testing data paths                                          </th><th>  </th></tr>
</thead>
<tbody>
<tr><td>recipes/testing/non_recipes/409d884a6896b8673a0643cb615a7d4b</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/7e36a2f244dbce34106e0b1c9b9b41ca</td><td>  </td></tr>
<tr><td>recipes/testing/recipes/955d93252cccff4b4d7943b8e678e367    </td><td>  </td></tr>
<tr><td>recipes/testing/recipes/63ec182e7c66babebcd4a2cc487b098b    </td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/b456e579f6cd6939e5013d250c7fe0ab</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/71eb580781257553cd6c850c8144492d</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/efdce7aa0c2c0c3c61f96ccc77e0f029</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/9edf96e7b7f8251c85c14f64e9df15d7</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/fd68ae7002749fe0e63a581fead0ff35</td><td>  </td></tr>
<tr><td>recipes/testing/non_recipes/9598041f7ffb1393f1a02f3f81d127d8</td><td>  </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m


Slaying daemons:


```python
factory.close_processes()
```


## Creating the Classifier
Now we have rendered the learning. Let's run some tests!

The classifier works as follows:

Given a function $z(d): W^u->[0,1]^v, \quad u\leq v$ a function to convert a document into a zipf where $d$ is a list of words and $W$ is the domain of possible words, a metric $m(P,Q): [0,1]^n \times [0,1]^m -> \mathbb{R}$, a learning set $L$ of $k$ tuples $(l_i, Z_i)$, where $l_i$ is the label of the set of zipfs $Z_i$, we proceed to classify a given document $d$ via two steps:

1. Convert the document d to zipf: $z_d = z(d)$
2. Predicted label is $l^* = \text{argmin}_{l_i} \left\{(l_i, Z_i): \frac{1}{\#\{Z_i\}}\sum_{z \in Z_i} m(z_d, z)\right\}$, where $\#\{Z_i\}$ is the cardinality of $Z_i$.


```python
def get_classifier_paths(dataset):
    """Return paths for classifier, given a dataset."""
    zipfs_path = get_build_paths(dataset)[2]
    learning_zipfs_path = "%s/learning"%zipfs_path
    testing_zipfs_path = "%s/testing"%zipfs_path
    return learning_zipfs_path, testing_zipfs_path
```


```python
def load_zipfs(classifier, path):
    """Load zipfs from given path into given classifier."""
    print("Loading zipfs from %s"%(b(path)))
    loaded = []
    for zipf in tqdm(get_zipfs(path), unit='zipf'):
        book_class = zipf.split('/')[-1].split('.')[0]
        args = zipf, book_class
        loaded.append(args)
        classifier.add_zipf(*args)
    
    random.seed(42)
    random.shuffle(loaded)
    display(HTML(tabulate.tabulate(loaded[:10], ["Path", "Class"], tablefmt='html')))
```


```python
def load_test_couples(path):
    """Return list of zipfs from given path."""
    print("Loading tests from %s"%(b(path)))
    test_couples = []
    for zipf in tqdm(get_zipfs(path)):
        book_class = zipf.split('/')[-2]
        args = zipf, book_class
        test_couples.append(args)

    random.seed(42)
    random.shuffle(test_couples)
    display(HTML(tabulate.tabulate(test_couples[:10], ["Path", "Class"], tablefmt='html')))
    return test_couples
```


```python
def metrics_test(classifier, test_couples):
    """Run test on all metrics usable on zipfs."""
    global metrics
    for metric in metrics:
        results = classifier.test(test_couples, metric)
        success(results, metric)
```

First we create the classifier with the options set above:


```python
classifier = ZipfClassifier(options)
print("We're using a classifier with options %s"%classifier)
```

    We're using a classifier with options {
        "sort": false
    }



```python
for dataset in datasets:
    print("Testing dataset %s"%(b(dataset)))
    learning_zipfs_path, testing_zipfs_path = get_classifier_paths(dataset)
    print(gray('-'*frame_number))
    load_zipfs(classifier, learning_zipfs_path)
    print(gray('-'*frame_number))
    test_couples = load_test_couples(testing_zipfs_path)
    print(gray('-'*frame_number))
    metrics_test(classifier, test_couples)
    classifier.clear()
    print(gray('='*frame_number))
```

    Testing dataset [1mauthors[0;0m
    I will build learning zipfs from [1mauthors/learning[0;0m,
    testing zipfs from [1mauthors/testing[0;0m
    and save them in [1mauthors/zipfs[0;0m
    
    [0;37m------------------------------[0;0m
    Loading zipfs from [1mauthors/zipfs/learning[0;0m



<table>
<thead>
<tr><th>Path                                   </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>authors/zipfs/learning/dh_lawrence.json</td><td>dh_lawrence</td></tr>
<tr><td>authors/zipfs/learning/twain.json      </td><td>twain      </td></tr>
<tr><td>authors/zipfs/learning/wilde.json      </td><td>wilde      </td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Loading tests from [1mauthors/zipfs/testing[0;0m



<table>
<thead>
<tr><th>Path                                             </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>authors/zipfs/testing/twain/double-barrelled.json</td><td>twain      </td></tr>
<tr><td>authors/zipfs/testing/twain/3294.json            </td><td>twain      </td></tr>
<tr><td>authors/zipfs/testing/wilde/2280.json            </td><td>wilde      </td></tr>
<tr><td>authors/zipfs/testing/dh_lawrence/3484.json      </td><td>dh_lawrence</td></tr>
<tr><td>authors/zipfs/testing/wilde/2299.json            </td><td>wilde      </td></tr>
<tr><td>authors/zipfs/testing/twain/3277.json            </td><td>twain      </td></tr>
<tr><td>authors/zipfs/testing/wilde/2318.json            </td><td>wilde      </td></tr>
<tr><td>authors/zipfs/testing/dh_lawrence/3487.json      </td><td>dh_lawrence</td></tr>
<tr><td>authors/zipfs/testing/wilde/2288.json            </td><td>wilde      </td></tr>
<tr><td>authors/zipfs/testing/twain/323.json             </td><td>twain      </td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Success with metric [0;33mjensen_shannon[0;0m: [1m79.38%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">127        </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 32        </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  1        </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  0.0180276</td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;"> 16        </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 16        </td></tr>
</tbody>
</table>


    Success with metric [0;33mnormal_total_variation[0;0m: [1m73.75%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">118        </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 42        </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  0.0265837</td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;"> 15        </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 21        </td></tr>
<tr><td>Mistook Dh_lawrence for Twain</td><td style="text-align: right;">  5        </td></tr>
<tr><td>Mistook Twain for Wilde      </td><td style="text-align: right;">  1        </td></tr>
</tbody>
</table>


    Success with metric [0;33mintersection_total_variation[0;0m: [1m78.75%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">126        </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 34        </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  0.0366234</td></tr>
<tr><td>Mistook Twain for Wilde      </td><td style="text-align: right;">  3        </td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;"> 10        </td></tr>
<tr><td>Mistook Dh_lawrence for Wilde</td><td style="text-align: right;">  4        </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 14        </td></tr>
<tr><td>Mistook Dh_lawrence for Twain</td><td style="text-align: right;">  2        </td></tr>
<tr><td>Mistook Twain for Dh_lawrence</td><td style="text-align: right;">  1        </td></tr>
</tbody>
</table>


    Success with metric [0;31mkullback_leibler[0;0m: [1m60.0%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">   Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">96       </td></tr>
<tr><td>failures                     </td><td style="text-align: right;">64       </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;"> 0       </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;"> 0.123656</td></tr>
<tr><td>Mistook Dh_lawrence for Wilde</td><td style="text-align: right;">26       </td></tr>
<tr><td>Mistook Twain for Wilde      </td><td style="text-align: right;">20       </td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;">13       </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 1       </td></tr>
<tr><td>Mistook Twain for Dh_lawrence</td><td style="text-align: right;"> 4       </td></tr>
</tbody>
</table>


    Success with metric [0;33mintersection_squared_hellinger[0;0m: [1m81.88%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">   Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">131      </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 29      </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  0      </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  2.99136</td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 20      </td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;">  6      </td></tr>
<tr><td>Mistook Dh_lawrence for Twain</td><td style="text-align: right;">  3      </td></tr>
</tbody>
</table>


    Success with metric [0;33mhellinger[0;0m: [1m81.25%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">130        </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 30        </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  0.0217585</td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;"> 15        </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 15        </td></tr>
</tbody>
</table>


    Success with metric [0;33msquared_hellinger[0;0m: [1m81.25%[0;0m



<table>
<thead>
<tr><th>Info                         </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                      </td><td style="text-align: right;">130        </td></tr>
<tr><td>failures                     </td><td style="text-align: right;"> 30        </td></tr>
<tr><td>unclassified                 </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                   </td><td style="text-align: right;">  0.0259997</td></tr>
<tr><td>Mistook Wilde for Dh_lawrence</td><td style="text-align: right;"> 15        </td></tr>
<tr><td>Mistook Wilde for Twain      </td><td style="text-align: right;"> 15        </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m
    Testing dataset [1mperiods[0;0m
    I will build learning zipfs from [1mperiods/learning[0;0m,
    testing zipfs from [1mperiods/testing[0;0m
    and save them in [1mperiods/zipfs[0;0m
    
    [0;37m------------------------------[0;0m
    Loading zipfs from [1mperiods/zipfs/learning[0;0m



<table>
<thead>
<tr><th>Path                                   </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>periods/zipfs/learning/romanticism.json</td><td>romanticism</td></tr>
<tr><td>periods/zipfs/learning/realism.json    </td><td>realism    </td></tr>
<tr><td>periods/zipfs/learning/naturalism.json </td><td>naturalism </td></tr>
<tr><td>periods/zipfs/learning/modernism.json  </td><td>modernism  </td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Loading tests from [1mperiods/zipfs/testing[0;0m



<table>
<thead>
<tr><th>Path                                                  </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>periods/zipfs/testing/romanticism/680.json            </td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/modernism/sea-and-sardinia.json </td><td>modernism  </td></tr>
<tr><td>periods/zipfs/testing/romanticism/158.json            </td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/romanticism/fugitive-pieces.json</td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/romanticism/3882.json           </td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/romanticism/129.json            </td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/modernism/misalliance.json      </td><td>modernism  </td></tr>
<tr><td>periods/zipfs/testing/romanticism/684.json            </td><td>romanticism</td></tr>
<tr><td>periods/zipfs/testing/modernism/in-the-cage.json      </td><td>modernism  </td></tr>
<tr><td>periods/zipfs/testing/realism/4137.json               </td><td>realism    </td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Success with metric [0;31mjensen_shannon[0;0m: [1m63.77%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">551         </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">313         </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0         </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.00737391</td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 16         </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;">137         </td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;"> 31         </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 25         </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;"> 11         </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;"> 27         </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;">  5         </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 16         </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;"> 10         </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 23         </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  3         </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;">  9         </td></tr>
</tbody>
</table>


    Success with metric [0;31mnormal_total_variation[0;0m: [1m58.8%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">508        </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">356        </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.0114856</td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 20        </td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;"> 27        </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;">133        </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 26        </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;"> 37        </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;"> 24        </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;"> 17        </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;">  8        </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 24        </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;"> 12        </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  3        </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 25        </td></tr>
</tbody>
</table>


    Success with metric [0;31mintersection_total_variation[0;0m: [1m59.49%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">514        </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">350        </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.0246552</td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;"> 40        </td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 30        </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;"> 85        </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;"> 23        </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 63        </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;"> 21        </td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;">  8        </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;"> 17        </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 33        </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  8        </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 20        </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;">  2        </td></tr>
</tbody>
</table>


    Success with metric [0;31mkullback_leibler[0;0m: [1m63.54%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">    Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">549       </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">315       </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0       </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.116207</td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 25       </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;"> 15       </td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;"> 33       </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 55       </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;"> 10       </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;"> 40       </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 41       </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;"> 34       </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;"> 26       </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;"> 15       </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 14       </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;">  7       </td></tr>
</tbody>
</table>


    Success with metric [0;32mintersection_squared_hellinger[0;0m: [1m87.5%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">   Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">756      </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">108      </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0      </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  1.19871</td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;">  8      </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 31      </td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 22      </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 15      </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  3      </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;"> 12      </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;">  3      </td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;">  2      </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;">  3      </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;">  5      </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;">  3      </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;">  1      </td></tr>
</tbody>
</table>


    Success with metric [0;31mhellinger[0;0m: [1m65.39%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">565         </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">299         </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0         </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.00946594</td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;"> 27         </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;">137         </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 24         </td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 10         </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;">  6         </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;"> 32         </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;">  4         </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 17         </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;">  7         </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 27         </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  4         </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;">  4         </td></tr>
</tbody>
</table>


    Success with metric [0;31msquared_hellinger[0;0m: [1m65.39%[0;0m



<table>
<thead>
<tr><th>Info                              </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                           </td><td style="text-align: right;">565        </td></tr>
<tr><td>failures                          </td><td style="text-align: right;">299        </td></tr>
<tr><td>unclassified                      </td><td style="text-align: right;">  0        </td></tr>
<tr><td>mean_delta                        </td><td style="text-align: right;">  0.0107078</td></tr>
<tr><td>Mistook Modernism for Realism     </td><td style="text-align: right;"> 27        </td></tr>
<tr><td>Mistook Romanticism for Realism   </td><td style="text-align: right;">137        </td></tr>
<tr><td>Mistook Realism for Romanticism   </td><td style="text-align: right;"> 24        </td></tr>
<tr><td>Mistook Modernism for Naturalism  </td><td style="text-align: right;"> 10        </td></tr>
<tr><td>Mistook Romanticism for Naturalism</td><td style="text-align: right;">  6        </td></tr>
<tr><td>Mistook Naturalism for Realism    </td><td style="text-align: right;"> 32        </td></tr>
<tr><td>Mistook Realism for Naturalism    </td><td style="text-align: right;">  4        </td></tr>
<tr><td>Mistook Naturalism for Modernism  </td><td style="text-align: right;"> 17        </td></tr>
<tr><td>Mistook Realism for Modernism     </td><td style="text-align: right;">  7        </td></tr>
<tr><td>Mistook Modernism for Romanticism </td><td style="text-align: right;"> 27        </td></tr>
<tr><td>Mistook Naturalism for Romanticism</td><td style="text-align: right;">  4        </td></tr>
<tr><td>Mistook Romanticism for Modernism </td><td style="text-align: right;">  4        </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m
    Testing dataset [1mrecipes[0;0m
    I will build learning zipfs from [1mrecipes/learning[0;0m,
    testing zipfs from [1mrecipes/testing[0;0m
    and save them in [1mrecipes/zipfs[0;0m
    
    [0;37m------------------------------[0;0m
    Loading zipfs from [1mrecipes/zipfs/learning[0;0m



<table>
<thead>
<tr><th>Path                                   </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>recipes/zipfs/learning/non_recipes.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/learning/recipes.json    </td><td>recipes    </td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Loading tests from [1mrecipes/zipfs/testing[0;0m



<table>
<thead>
<tr><th>Path                                                                   </th><th>Class      </th></tr>
</thead>
<tbody>
<tr><td>recipes/zipfs/testing/non_recipes/3d42a56c681d56a73985877ba3c8bd6f.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/7c3d10eb81a2660e81cff3c3b41a951e.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/recipes/e2cd675ec121020b224e5118b5ae4d1b.json    </td><td>recipes    </td></tr>
<tr><td>recipes/zipfs/testing/recipes/2a90bc4cd74f6bbc578c9c5d8fda741f.json    </td><td>recipes    </td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/bfbc1b5eee1d423942b8b0b55f82bbff.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/6c41a532214cc6dfdbcc9b3bb5be0630.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/c31a242182356ea14acf080f628b03cc.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/bbecd5cafac80b5441765704a18db2c3.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/3a2b4e9cb1130e371f12e48525e2e190.json</td><td>non_recipes</td></tr>
<tr><td>recipes/zipfs/testing/non_recipes/38928442e96835f1897ab5c1cb4276a3.json</td><td>non_recipes</td></tr>
</tbody>
</table>


    [0;37m------------------------------[0;0m
    Success with metric [0;32mjensen_shannon[0;0m: [1m85.71%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">6183        </td></tr>
<tr><td>failures                       </td><td style="text-align: right;">1030        </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   1        </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.0484755</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;">1030        </td></tr>
</tbody>
</table>


    Success with metric [0;33mnormal_total_variation[0;0m: [1m72.04%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">5197        </td></tr>
<tr><td>failures                       </td><td style="text-align: right;">2016        </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   1        </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.0538705</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;">2016        </td></tr>
</tbody>
</table>


    Success with metric [0;32mintersection_total_variation[0;0m: [1m93.64%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">6755        </td></tr>
<tr><td>failures                       </td><td style="text-align: right;"> 459        </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   0        </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.0799341</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;"> 458        </td></tr>
<tr><td>Mistook Recipes for Non_recipes</td><td style="text-align: right;">   1        </td></tr>
</tbody>
</table>


    Success with metric [0;31mkullback_leibler[0;0m: [1m28.64%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">     Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">2066       </td></tr>
<tr><td>failures                       </td><td style="text-align: right;">5148       </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   0       </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.848787</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;">5148       </td></tr>
</tbody>
</table>


    Success with metric [0;32mintersection_squared_hellinger[0;0m: [1m99.79%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">    Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">7199      </td></tr>
<tr><td>failures                       </td><td style="text-align: right;">  15      </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   0      </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   6.90183</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;">  14      </td></tr>
<tr><td>Mistook Recipes for Non_recipes</td><td style="text-align: right;">   1      </td></tr>
</tbody>
</table>


    Success with metric [0;32mhellinger[0;0m: [1m92.47%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">6671        </td></tr>
<tr><td>failures                       </td><td style="text-align: right;"> 541        </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   2        </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.0540858</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;"> 541        </td></tr>
</tbody>
</table>


    Success with metric [0;32msquared_hellinger[0;0m: [1m92.49%[0;0m



<table>
<thead>
<tr><th>Info                           </th><th style="text-align: right;">      Values</th></tr>
</thead>
<tbody>
<tr><td>success                        </td><td style="text-align: right;">6672        </td></tr>
<tr><td>failures                       </td><td style="text-align: right;"> 541        </td></tr>
<tr><td>unclassified                   </td><td style="text-align: right;">   1        </td></tr>
<tr><td>mean_delta                     </td><td style="text-align: right;">   0.0821632</td></tr>
<tr><td>Mistook Non_recipes for Recipes</td><td style="text-align: right;"> 541        </td></tr>
</tbody>
</table>


    [0;37m==============================[0;0m


## Root of errors

Here follows a list of some of the possible cause of errors I have diagnosed in the dataset formulation:

### Cardinality of difference of sets
The difference of the two sets has an important effect on the good result of the classification: if $\#\{P \setminus P \cap Q\} >> \#\{Q \setminus P \cap Q\}$ the metric should include only the intersection. It is for this reason that the `intersection_squared_hellinger` metric works best generally, but expecially in these situation such in the case of datasets **periods**.

### Small texts
When a text is significantly smaller than the average element in the learning set it will only be marked as a false positive or negative. In these datasets I have removed elements with less than 200 characters for this reason, since they do not offer enough informations for a significant classification.

## Conclusions
The classification method proposed, expecially using the `intersection_squared_hellinger` is extremely fast: in average it converts to zipf an average recepy and test it in **1.06 ms ± 50.3 µs**. It also is, in the case of web articles in particular, in average, correct and consistent: it could be used as a fast catalogation tecnique for focused web crawlers as a part of the filters to remove unwanted content. Combinations of the distances proposed might bring an higher success rate.

### Future works
In the near future, I'll develop the classifier using a learning algoritms to determine which combinations of distances brings the best success rate. Also, I'll be trying to use this classifier as a way to power an autonomous crawler starting from the current implementation of an other project of mine [TinyCrawler](https://github.com/LucaCappelletti94/tinycrawler).
