# Project Proposal

## AI Meta-Orchestrator for Multi-Model Task Routing

### System Architecture (cute)
<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/323d3031-e1b4-4c0a-b3f5-2a3693cd91c6" />

### System Architecture (for real)
<img width="971" height="676" alt="image" src="https://github.com/user-attachments/assets/aa75f47c-f017-4ee3-b621-29f716aa50c0" />


## Team

- Team member 1: Gillhofer Julia
- Team member 2: Tischler Michel 

## Project Repository

- Project repository: https://github.com/kis4-2026ss/projekt-g3-gillhofer-tischler
- Reference implementations:
  - LangGraph
  - OpenRouter
  - LiteLLM
  - AutoGen
  - ...

---

## 1. Goal of the Project

### High-Level Goal

The goal of this project is to design and implement an AI orchestration system that can analyze complex user requests, decompose them into smaller subtasks, automatically select the most suitable AI models or tools for each subtask, and merge the generated outputs into one coherent final solution.

The system acts as a meta-layer above existing AI systems such as GPT, Claude, Gemini, local models, and specialized APIs.

### Validation of the Goal

We will validate the project using the following criteria :

- The system accepts a complex user task as input.
- The task is automatically decomposed into meaningful subtasks.
- Models can be routed dynamically.
- Results from all subtasks are aggregated into a final output.
- The system performs automated quality validation before output generation.
- Routing decisions and execution logs are traceable.

We will additionally validate the project by comparing our system with a common LLM.
Both systems will receive a complex Task and we will determine which output is better. 

---

## 2. System, Feature, or Workflow to be Developed

We will develop an AI orchestration platform capable of intelligent model routing and multi-agent execution.

### Main System Components

- **Task Analyzer**: Interprets user intent and extracts requirements.
- **Task Decomposer**: Splits complex tasks into executable subtasks.
- **Capability Registry**: Stores model capabilities, cost, latency, and specialization.
- **Model Router**: Selects the best AI system for each subtask.
- **Execution Layer**: Calls external APIs or local models.
- **Result Aggregator**: Combines outputs into a unified solution.
- **Quality Validator**: Reviews consistency, correctness, and completeness.
- **Observability Layer**: Logs decisions, performance, costs, and errors.

### Development Workflow to be Analyzed

The project will analyze the following AI-assisted workflow:

1. Requirement analysis
2. Task decomposition
3. Capability matching
4. Multi-model execution
5. Output validation
6. Final result synthesis

---

## 3. AI Tools and Models

We plan to use the following tools:

- GPT models for reasoning and structured synthesis
- Claude models for code generation and long-context analysis
- Gemini models for research and summarization
- (Local models) for structured extraction and privacy-sensitive tasks
- LangGraph for workflow orchestration
- LiteLLM for unified model access

### AI Contribution by Development Stage

#### Stage 1: Architecture Planning

AI tools will support system design, architecture proposals, and interface definitions.

#### Stage 2: Manual Model Research

AI tools will support selecting a few (researched by us) models, to implement the workflow whithout using an automatic routing of the models. 

#### Stage 3: Prototype Development

AI tools will support implementation of orchestration pipelines and integration code.

#### Stage 4: Evaluation and Optimization

AI tools will support benchmark generation and failure analysis.

#### Stage 5: Routing Logic

AI tools will support switching from hand-picked models to automatic routing.

#### Stage 6: Documentation and Reflection

AI tools will support summarization, architecture explanation, and documentation generation.

---

## 4. Project Plan

The project is planned for approximately 25 hours of total effort.
The deadlines include the day by which they must be met.

### Task 1: Research and Architecture Design

**Responsible:** Both team members |
**Internal deadline:** 15.05.2026 |
**Estimated effort:** 2 hours per Person

Activities:

- Analyze existing orchestration frameworks.
- Define target architecture.

### Task 2: Infrastructure Setup

**Responsible:** Both team members |
**Internal deadline:** 19.05.2026 |
**Estimated effort:** 3 hours per Person

Activities:

- Setup orchestration framework.
- Configure  Main-LLM.
- Connect first model providers (manually selected models).

### Task 3: Task Analyzer and Decomposer

**Responsible:** Both team members |
**Internal deadline:** 22.05.2026 |
**Estimated effort:** 2 hours per Person

Activities:

- Implement input parsing.
- Implement task decomposition logic.
- Create task schema.

### Task 4: Aggregation and Validation Layer

**Responsible:** Both team members |
**Internal deadline:** 24.05.2026 |
**Estimated effort:** 2 hours per Person

Activities:

- Merge model outputs.
- Implement consistency checks.
- Add quality scoring.

### Task 5: Dynamic Model Router

**Responsible:** Both team members |
**Internal deadline:** 26.05.2026 |
**Estimated effort:** 2 hours per Person

Activities:

- Implement capability registry.
- Build routing logic.
- Define scoring mechanisms.

### Task 6: Testing and Documentation

**Responsible:** Both team members |
**Internal deadline:** 29.05.2026 |
**Estimated effort:** 1 hour per Person

Activities:

- Document routing examples.
- Prepare final presentation.



