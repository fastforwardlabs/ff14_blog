---
toc: true
layout: post
description: Introduction to NLP for QA
categories: [markdown]
title: An Example Markdown Post take TWO
---
# NLP for Automated Question Answering


![]({{ site.baseurl }}/images/post1/reading_retriever.png "Get it? Retriever
? Reader?")

## Information Retrieval-Based Systems: Retrievers and Readers

Information retrieval-based question answering (IR QA) systems find and extract 
a text segment from a large collection of documents. The collection can be as 
vast as the entire web (open domain) or as specific as a company’s Confluence 
documents (closed domain). Contemporary IR QA systems first identify the most 
relevant documents in the collection, and then extract the answer from the 
contents of those documents. To illustrate this approach, let’s revisit our 
Google example from the introduction, only this time we’ll include some of the 
search results!

![]({{ site.baseurl }}/images/post1/abe_search.png "Did Abe have big
 ears?")


We already talked about how the snippet box acts like a QA system. 
The search results below the snippet illustrate some of the reasons why an IR QA 
system can be more useful than a search engine alone. The relevant links vary 
from essentially advertising (study.com), to making fun of Lincoln’s ears 
(Reddit at its finest), to a discussion of color blindness (answers.com without 
the answer we want), to an article about all presidents’ eye color (getting 
warmer, Chicago Tribune), to the last link, answers.yahoo.com, which is on-topic, 
and narrowly scoped to Lincoln but gives an ambiguous answer. Without the 
snippet box at the top, a user would have to skim each of these links looking 
for their answer. 

IR QA systems are not just search engines, which take general natural language 
terms and provide a list of relevant documents. IR QA systems perform an 
additional layer of processing on the most relevant documents to deliver a 
pointed answer based on the contents of those documents (like the snippet box). 
While we won’t hazard a guess at exactly how Google extracted “gray” from these 
search results, we can examine how an IR QA system could exhibit similar 
functionality in a real world (e.g., non-Google) implementation. Below we 
illustrate the workflow of a generic IR-based QA system. These systems generally 
have two main components: the document retriever and the document reader.  

![]({{ site.baseurl }}/images/post1/QAworkflow.png "Generic IR QA
 system")


The document retriever functions as the search engine, ranking and retrieving 
relevant documents to which it has access. It supplies a set of candidate 
documents that could answer the question (often with mixed results, per the 
Google search shown above). The second component is the document reader: 
reading comprehension algorithms built with core NLP techniques. This component 
processes the candidate documents and extracts from one of them an explicit span 
of text that best satisfies the query. Let’s dive into each of these components. 