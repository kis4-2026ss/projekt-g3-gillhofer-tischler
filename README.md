[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kz4Hl53G)
# AI Meta-Orchestrator: Multi-Model Task Routing Engine

A high-performance, autonomous orchestration system that decomposes complex user requests, routes subtasks to specialized "best-in-class" free AI models, and synthesizes a verified final output.

## 🚀 Key Capabilities

*   **Intelligent Task Decomposition**: Automatically breaks down complex, multi-part requests (e.g., "Research X, write code for Y, and generate an image prompt for Z") into logical subtasks.
*   **Parallel Map-Reduce Execution**: Executes independent subtasks concurrently using a high-speed parallel architecture, significantly reducing total response time.
*   **Quality-Based Routing**: Dynamically selects the best model for every specific task (Coding, Reasoning, Creativity, Research) based on granular performance benchmarks rather than just speed.
*   **Automated Code Verification**: Includes a built-in "Senior Engineer" audit node that scans generated code for syntax errors and logical flaws.
*   **Multimodal Handling**: Supports image analysis and provides professional-grade "Master Prompts" for image generation when direct API access is limited.
*   **Transparent Insights**: Provides a detailed breakdown of which models were used and the rationale behind every routing decision.

## 🛠 How It Works (The Architecture)

The system is built on **LangGraph** and uses a **State-Machine** approach to manage the lifecycle of a request:

1.  **Task Analyzer**: Uses a reasoning-heavy model to identify user intent and complexity. It employs a "Deep Research" pattern for complex queries (Find Facts -> Analyze -> Verify).
2.  **Model Router**: Consults the **Capability Registry** (a database of 25+ free models) to match subtasks with the highest-scoring models for that specific modality.
3.  **Parallel Executor**: Utilizing the **LangGraph Send API**, the system spawns multiple worker nodes to execute independent subtasks simultaneously.
4.  **Code Verifier**: A specialized node that performs a secondary audit on any subtask that produced programming code.
5.  **Result Aggregator**: A synthesis engine that combines all subtask outputs, highlights local files/images, and ensures a cohesive final response.
6.  **Quality Validator**: A final gatekeeper that scores the output (0-100) on consistency, correctness, and completeness.

## 💡 Why & When to Use It

### **Why use this over a single LLM (like ChatGPT/Claude)?**
*   **Specialization**: No single model is the best at everything. This app uses a "Coding specialist" for code and a "Creative specialist" for prose within the same request.
*   **Cost Efficiency**: It exclusively utilizes "Free Tier" models from OpenRouter, providing premium-level results without subscription costs.
*   **Reliability**: Automated verification and multi-model rotation ensure that if one model fails or is rate-limited, the system automatically tries another.

### **When to use it?**
*   **Complex Projects**: When you have a task with multiple distinct steps.
*   **High-Stakes Coding**: When you want an extra layer of verification on generated scripts.
*   **Comparative Research**: When you need to gather information from multiple perspectives or sources.

## 📂 Project Structure

*   `main.py`: The entry point for the application.
*   `src/orchestrator/graph.py`: The core LangGraph orchestration logic.
*   `src/orchestrator/nodes/`: Specialized logic for each phase (Analyzer, Router, Executor, Verifier, Aggregator, Validator).
*   `src/orchestrator/nodes/registry.py`: The intelligence engine that selects models based on quality benchmarks.
*   `outputs/`: Local directory where generated images and downloaded assets are stored.
*   `scripts/update_free_models.py`: A utility to refresh the registry with the latest available free models and performance scores.

## ⚙️ Setup & Usage

1.  **Requirements**: Python 3.10+, LiteLLM, LangGraph.
2.  **Environment**: Add your `OPENROUTER_API_KEY` to a `.env` file.
3.  **Run**:
    ```bash
    python main.py "Your complex request here"
    ```
4.  **Update Models**:
    ```bash
    python scripts/update_free_models.py
    ```
