# Product Guidelines - sd-optim

## Engineering Standards
- **Technical Excellence:** Documentation and comments must be technical, direct, and focused on architectural clarity.
- **Strict Typing:** Mandatory type hinting for all new and refactored code to ensure long-term maintainability.
- **Architectural Integrity:** Maintain a decoupled architecture where core components like scorers and optimizers are isolated from the main execution loop.

## Operational Guidelines
- **Enhanced Telemetry:** 
    - Use structured logging to facilitate debugging and analysis.
    - Provide granular verbosity control (Minimal, Standard, Developer).
    - Capture and log performance metrics for merging, generation, and scoring phases to monitor system health and efficiency.
- **Configuration Safety:** Leverage Hydra's schema validation to enforce strict configuration integrity before the system starts.

## Design Philosophy (Dashboard & UI)
- **Data-First Design:** Prioritize high-fidelity data visualization, including detailed charts and performance tables, over aesthetic simplicity. The focus is on providing deep insights into the optimization landscape.
