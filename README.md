# minitorch

[MiniTorch](https://minitorch.github.io/) is a DIY teaching library for machine learning engineers who wish to learn about the internal concepts underlying deep learning systems. 
It is a pure Python re-implementation of the [Torch](https://pytorch.org/) API designed to be simple, easy to read, tested, and incremental. 
The final library can run Torch code.

Individual assignments cover:
- ML Programming Foundations
- Autodifferentiation
- Tensors
- GPUs and Parallel Programming
- Foundational Deep Learning

The project was developed by [Sasha Rush](http://rush-nlp.com/) ([@srush_nlp](https://twitter.com/srush_nlp)) with Ge Gao, Anton Abilov, and Aaron Gokaslan.

## Task 0.5 Visualization

While testing is excellent for maintaining correctness, exploratory analysis is critical for gaining intuition. When you are stuck, often the best thing to do is to look at your data and outputs. Visualizing our system can't prove that it is correct, but it can often directly help us to figure out what goes wrong. 

Throughout our development, we will use visualization to observe intermediate states, training progress, outputs, and even final models.
The main library we will use is called Streamlit ([streamlit/streamlit](https://github.com/streamlit/streamlit)).

For example, we can hand-create classifiers that split the linear dataset into the correct colors by dragging a slider bar:
![Screenshot 2023-03-09 at 22-11-51 interactive minitorch](https://user-images.githubusercontent.com/43287234/224052620-7ef8976e-c741-4757-9762-64b6fce2e76a.png)
![Screenshot 2023-03-09 at 22-23-07 Editing minitorch_README md at main · eatPizza311_minitorch · GitHub](https://user-images.githubusercontent.com/43287234/224053960-fc17b9ce-2784-473b-a09d-b953cf6dc066.png)

Or check whether our implementation of `Module` class has the correct tree data structure:
![Screenshot 2023-03-08 at 22-52-18 interactive minitorch](https://user-images.githubusercontent.com/43287234/224054147-16bb1b16-ef3c-4b99-91cd-ddd9fcc9dc9d.png)
