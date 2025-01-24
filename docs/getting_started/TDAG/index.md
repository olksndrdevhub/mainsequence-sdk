# Welcome to TDAG

**TDAG**  is a cutting-edge, graph-based library designed specifically for building and managing **time-series data pipelines**. With TDAG, you can create automated, time-based dependency structures that are robust, efficient, and ready for real-world, scalable applications.

At its core, **TDAG** leverages the power of **DAGs** (Directed Acyclic Graphs). A DAG is a graph with nodes connected by edges, where the edges have a direction, and no cycles (loops) exist. In simpler terms, this structure allows data to flow in one direction without any feedback loops, which is essential for building reliable and predictable data pipelines. The "time-directed" aspect in TDAG makes it ideal for handling time-sensitive operations, ensuring that tasks occur in the correct sequence.

### Why TDAG?
With TDAG, you can automatically create **time-based data pipelines** that handle complex dependencies. The library provides features such as automatic hashing, seamless scheduling integration, and a structured approach that enhances the reliability and scalability of your pipelines.

### Key Features:
- **Automated Time-Based Pipelines**: TDAG allows you to easily build data pipelines where tasks are executed in time order, respecting dependencies.
- **Built-in Scheduling and Hashing**: Pipelines are automatically hashed and scheduled for efficient execution.
- **Scalable & Robust**: Whether you're working on small datasets or massive time-series data flows, TDAG scales to meet your needs while ensuring the entire process is fault-tolerant and robust.

### Use Case: Investment Strategies and Beyond
One of TDAG's main use cases is transforming raw financial data into actionable insights, such as investment strategy predictions or portfolio weights. TDAG simplifies the process of managing complex time-based operations in **financial modeling**, helping you move from data to decisions effortlessly.

However, **TDAG** is not just limited to finance! It's perfect for **any application requiring time-sensitive data pipelines**, particularly in **live and online modes** where real-time decision-making is crucial. For example, TDAG can be used in **online training of machine learning models**, where time-based data flow and immediate processing are essential for model accuracy and performance.

---

### Why Use a DAG?

A **DAG (Directed Acyclic Graph)** is a graph structure where:
1. **Directed**: Each connection (edge) between nodes points in a specific direction, indicating the flow of data or dependencies.
2. **Acyclic**: There are no cycles, meaning that no node in the graph can loop back to itself. This is critical for tasks that need to happen in a specific sequence.

In the context of **TDAG**, a DAG ensures that all data processing happens in the correct order, and no task is repeated or stuck in a loop. When applied to time-series data pipelines, this means that your data will always flow from the past to the present in a structured, predictable manner, ensuring that dependencies are handled properly and efficiently.

---


### The Power of TDAG
Whether you’re managing financial data pipelines or real-time machine learning workflows, TDAG is designed to give you the **control**, **scalability**, and **reliability** you need to handle complex, time-sensitive data with ease.

Start by exploring our [Getting Started Tutorial](tutorial/getting_started/getting_started.md) or jump into the [Code Reference](reference) if you're already familiar with TDAG.

If you are looking for more resources you can also access our vidoe tutorials on TDAG here:

* [Main Sequence SDK Tutorial:  1 Introduction to TDAG](https://www.loom.com/share/046d733baf2e4ee2ba185c14717bc576?sid=2e934ed3-85b4-4c3a-a8ef-63cfdc440b0b)
* [Main Sequence SDK Tutorial:  2 Introduction to TimeSeries](https://www.loom.com/share/bb0935c7d05d41ad91b921681f5c7631?sid=4ed62862-4a36-46d4-977b-6d7379bdb33f)
* [Main Sequence SDK Tutorial:  3 Introduction to Schedulers](https://www.loom.com/share/a27106d450d84509ad879422aed09219?sid=52f9c27d-4498-4779-a85e-f46597c03019)

