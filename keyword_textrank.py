import itertools
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def textrank_keyword_extraction(text, window_size=4, num_keywords=10):
    # Tokenize and filter stopwords
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Construct a graph
    graph = nx.Graph()
    graph.add_nodes_from(set(words))
    
    # Add edges between words that co-occur within the window size in the text
    for i in range(len(words) - window_size + 1):
        window_words = words[i: i + window_size]
        for pair in itertools.combinations(window_words, 2):
            if pair[0] != pair[1]:
                graph.add_edge(*pair)

    # Apply PageRank algorithm to the graph
    ranks = nx.pagerank(graph)

    # Sort words by rank and extract keywords
    ranked_words = sorted(((ranks[word], word) for word in ranks), reverse=True)
    keywords = [word for _, word in ranked_words[:num_keywords]]

    return keywords

# Example Usage
text = """
LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. 
Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, 
the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, 
suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. 
"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," 
currently six places below his number one movie on the UK box office chart. 
Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. 
"Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. 
Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. 
"But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" 
is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter's latest » . 
There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. 
He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." 
Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . 
Copyright 2007 Reuters. All rights reserved.
This material may not be published, broadcast, rewritten, or redistributed.
"""
# Replace with your actual text
keywords = textrank_keyword_extraction(text)
print(keywords)
