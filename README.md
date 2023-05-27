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


## What's the difference?
🚨 This repository is using the current state (2023/04) of the original minitorch with several modifications. 🚨

Here are what I've changed:
- Use [Ruff](https://github.com/charliermarsh/ruff) to replace [flake8](https://github.com/PyCQA/flake8), [isort](https://github.com/PyCQA/isort), and [darglint](https://github.com/terrencepreilly/darglint).

   This is mainly because the owner archived [darglint](https://github.com/terrencepreilly/darglint) on Dec 16, 2022.
   
   And Ruff has a [plan to implement the functionality](https://github.com/charliermarsh/ruff/issues/458), so I decide to replace flake8 and isort at once.
   
   I add the `.pre-commit-config.yaml` and `ruff.toml` that allows us to utilize [pre-commit](https://pre-commit.com/) base on `setup.cfg`.
   
   But note that Ruff does not support some rules, so that you won't find them in the abovementioned Ruff configuration file.
   
   Here is the list of not supported rules:

    - [E203Whitespace before ':'](https://www.flake8rules.com/rules/E203.html)
    - [E266Too many leading '#' for block comment](https://www.flake8rules.com/rules/E266.html)
    - [W503Line break occurred before a binary operator](https://www.flake8rules.com/rules/W503.html)
    - [F812List comprehension redefines name from line n](https://www.flake8rules.com/rules/F812.html)
      
- I commented out all the code in `mlprimer.py` because it seems incomplete and will affect GitHub Actions.

## Task 0.5 Visualization

While testing is excellent for maintaining correctness, exploratory analysis is critical for gaining intuition. When you are stuck, often the best thing to do is to look at your data and outputs. Visualizing our system can't prove that it is correct, but it can often directly help us to figure out what goes wrong. 

Throughout our development, we will use visualization to observe intermediate states, training progress, outputs, and even final models.
The main library we will use is called Streamlit ([streamlit/streamlit](https://github.com/streamlit/streamlit)).

For example, we can hand-create classifiers that split the linear dataset into the correct colors by dragging a slider bar:
![Screenshot 2023-03-09 at 22-11-51 interactive minitorch](https://user-images.githubusercontent.com/43287234/224052620-7ef8976e-c741-4757-9762-64b6fce2e76a.png)
![Screenshot 2023-03-09 at 22-23-07 Editing minitorch_README md at main · eatPizza311_minitorch · GitHub](https://user-images.githubusercontent.com/43287234/224053960-fc17b9ce-2784-473b-a09d-b953cf6dc066.png)

Or check whether our implementation of `Module` class has the correct tree data structure:
![Screenshot 2023-03-08 at 22-52-18 interactive minitorch](https://user-images.githubusercontent.com/43287234/224054147-16bb1b16-ef3c-4b99-91cd-ddd9fcc9dc9d.png)
