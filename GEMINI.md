# AI Meta-Orchestrator: GEMINI.md

This document serves as the foundational instruction set for the development of the AI Meta-Orchestrator. It aligns the technical execution with the project proposal.

## 1. Project Identity & Goal
- **Project Name**: AI Meta-Orchestrator for Multi-Model Task Routing
- **Team**: Gillhofer Julia, Tischler Michel
- **Core Goal**: Build an orchestration system that decomposes complex user requests, routes subtasks to the most suitable AI models, and synthesizes a validated final output.

## 2. System Architecture
The system is composed of the following core modules:
- **Task Analyzer**: Interprets user intent and requirements.
- **Task Decomposer**: Splits complex tasks into executable subtasks.
- **Capability Registry**: Metadata store for model capabilities, cost, latency, and specializations.
- **Model Router**: The decision engine for dynamic model selection.
- **Execution Layer**: Handles external API (via LiteLLM) and local model calls.
- **Result Aggregator**: Synthesizes subtask outputs into a coherent whole.
- **Quality Validator**: Performs consistency, correctness, and completeness reviews.
- **Observability Layer**: Logs decisions, performance, and costs for traceability.

## 3. Tech Stack & Integration
- **Orchestration**: LangGraph
- **Unified Model Access**: LiteLLM
- **Models**: GPT (Reasoning), Claude (Code/Long-context), Gemini (Research), Local Models (Privacy/Structure).
- **Environment**: Python-based ecosystem.

## 4. Development Standards
- **Modular Design**: Each component (Analyzer, Decomposer, etc.) must be isolated and testable.
- **Traceability**: Every routing decision MUST be logged in the Observability Layer with its rationale.
- **Validation-Driven**: No output is final until it passes the Quality Validator.
- **Type Safety**: Use Pydantic/Type hints for task schemas and model metadata.

## 5. Implementation Roadmap (Reference)
1. **Infrastructure**: Setup LangGraph and LiteLLM integration.
2. **Analysis/Decomposition**: Focus on intent extraction and subtask schema.
3. **Synthesis/Validation**: Implement aggregation and quality scoring.
4. **Dynamic Routing**: Transition from hand-picked models to automated registry-based routing.

## 6. Verification Criteria
- Task decomposition into meaningful units.
- Dynamic routing based on subtask requirements.
- Aggregated output consistency.
- Automated quality validation.
- Comparative benchmarks against single-model LLM outputs.

## 8. Image Generation Handling
- **Local Storage**: All generated images are automatically downloaded and saved to the `outputs/` directory.
- **Fallback Logic**: If no suitable image generation model is available (e.g., in free-only mode), the system will provide a "Professional Image Generation Prompt" block in the final output.
- **Capability Routing**: Use the `image_generation` capability for text-to-image tasks.
