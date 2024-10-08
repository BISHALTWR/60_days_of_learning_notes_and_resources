import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n_pages = len(corpus)
    probabilities = dict.fromkeys(corpus.keys(), (1-damping_factor)/n_pages)

    links = corpus[page]
    if links:
        linked_prob = damping_factor/len(links)
        for linked_page in links:
            probabilities[linked_page] += linked_prob
    else:
        for prob in probabilities:
            probabilities[prob] += damping_factor/n_pages
    
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # raise NotImplementedError
    pages = list(corpus.keys())
    current_page = random.choice(pages)
    pagerank = dict.fromkeys(pages, 0)

    for _ in range(n):
        pagerank[current_page] += 1
        probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(probabilities.keys()), list(probabilities.values()), k=1)[0]
    
    for page in pagerank:
        pagerank[page] /= n
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # raise NotImplementedError
    n_pages = len(corpus)
    pagerank = dict.fromkeys(corpus.keys(), 1/n_pages)
    done = False

    while not done:
        new_pagerank = {}
        for page in corpus:
            new_rank = (1-damping_factor)/n_pages
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    new_rank += damping_factor * pagerank[linking_page]/len(corpus[linking_page])
                if not corpus[linking_page]:
                    new_rank += damping_factor * pagerank[linking_page]/n_pages
                
            new_pagerank[page] = new_rank
        done = all(abs(new_pagerank[page] - pagerank[page]) <= 0.001 for page in pagerank) #this is fun
            
        pagerank = new_pagerank
    return pagerank


if __name__ == "__main__":
    main()
