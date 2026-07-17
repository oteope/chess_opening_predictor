# ♟️ Chess Opening Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **How much information about the final outcome of a chess game is already contained in the opening?**

<p align="center">
    <img src="assets/images/chess pipeline.png" alt="Chess Opening Predictor Pipeline" width="100%">
</p>

## 📖 Project Overview

Chess engines evaluate millions of positions every second using handcrafted heuristics and deep search algorithms.

This project deliberately takes a different direction.

Instead of building another chess engine, it investigates how much predictive information is already contained in the opening phase of a chess game.

Each game is represented only by:

- the board position after move 10,
- White Elo,
- Black Elo,
- Elo Difference.

No chess heuristics, engine evaluations or handcrafted positional features are provided to the models.

The project benchmarks a classical Machine Learning approach (Random Forest) against a Deep Learning approach (Multi-Layer Perceptron implemented in PyTorch) under identical experimental conditions.

## 🎯 Research Question

This project aims to answer a simple but meaningful research question:

> **How much predictive information about the final outcome of a chess game is already contained in the opening?**

More specifically, the project investigates:

- Can the geometry of the board after move 10 predict the final result?
- How much predictive power comes from player strength alone?
- Does combining board geometry with Elo ratings improve prediction?

## 💡 Motivation

Modern chess engines achieve outstanding performance by combining handcrafted evaluation functions with powerful search algorithms.

This project intentionally avoids that approach.

Instead of relying on expert chess knowledge, the models receive only a raw representation of the board together with the players' ratings.

By removing manually engineered chess heuristics, the project focuses entirely on the ability of Machine Learning algorithms to discover useful patterns directly from historical data.

The objective is not to create a stronger chess engine, but to study the predictive information already present in chess openings.
