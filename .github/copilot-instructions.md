# Repository guidance

## Build, run, and checks

- Create the application environment with `conda env create -f environment.yml` (Python 3.12), then activate it with `conda activate shoreline_monitor_wps`.
- Copy `configuration.txt.example` to `configuration.txt` and configure its `[database]` PostgreSQL settings before running database-backed processes.
- Start the local Flask/PyWPS service from the repository root with `python pywps.wsgi`.
- There is no configured build, test, or lint runner and no checked-in test suite, so no single-test command is currently available.

## Architecture

- `pywps.wsgi` creates the Flask application, registers the PyWPS processes, and initializes the `/feedback` endpoint. It also serves generated files from `data/` at `/data/`.
- Each WPS process is a `pywps.app.Process` adapter in `processes/wps_*.py`. Shoreline adapters parse the JSON `profileid` input, call the corresponding handler in `shoreline_getprofile.py` or `shoreline_getstats.py`, and return a URL to a generated HTML result.
- The shoreline handlers query the PostGIS `gctr` and `shorelinemonitor_series` tables and use `generate_dynamic_plot.py` to build interactive Panel/HoloViews visualizations. The profile process uses `gctr.index` as its profile ID and finds nearby transects; the statistics process summarizes a 5 km neighborhood. Results are written to `data/`.
- `processes/feedback.py` persists feedback to `public.shoreline_feedback`; its table definition is in `scripts/05_create_feedback_table.sql`. The `scripts/` directory also contains data-ingestion and database-preparation utilities, including Azure Parquet-to-PostGIS ingestion.

## Repository conventions

- To expose a WPS process, implement a `Process` adapter and add an instance to the `processes` list in `pywps.wsgi`. Keep its WPS `identifier`, input/output identifiers, and supported MIME formats aligned with clients and XML examples.
- WPS shoreline inputs are complex `application/json` values. The adapters decode the input data, call a handler, then serialize its result into the declared output; keep plot/data work in the handler modules rather than Flask routes.
- `profileid` means the `gctr.index` value, not `transect_id`. Queries and the client-facing process contract depend on that distinction.
- `/feedback` ignores transect metadata sent by the client. It looks the metadata up in `gctr` by `profileid`, verifies `transect_id` matches, and stores the feedback without metadata if it doesn't. Change-rate class labels come from `processes/classification.py`, shared with the profile plot.
- Database-backed application settings live in the untracked root `configuration.txt`, using the `[database]` keys shown in `configuration.txt.example`. Azure ingestion scripts read storage settings from environment variables (see `.env.example`).
- Spatial neighborhood queries use the pre-transformed `gctr.geom_3857` geometry in EPSG:3857, where distances such as the 5,000 m radius are in meters. Preserve the expected PostGIS geometry and table column names when changing queries or ingestion.
- Generated profile plots use the `plot_` filename prefix and statistics plots use `stat_`; both are published from `data/` and linked through the configured service URL.



## Copilot workflow and response preferences

### Planning and implementation
- For significant changes or requests involving a plan, explain the proposed approach before implementation.
- Identify the files and components expected to change and describe the purpose of each change.
- Highlight relevant risks, dependencies, and testing requirements.
- Do not implement changes when I explicitly ask for investigation or analysis only.
- Avoid unrelated refactoring and unnecessary changes.

### Code explanations
- When explaining how something works, include relevant code snippets from the codebase.
- Trace the execution flow step by step, showing how the relevant functions, modules, and components interact.
- Explain technical concepts in plain language before introducing jargon.
- Use concise tables when comparing components, responsibilities, or alternatives.

### Frontend and browser testing
- When changing frontend code or UI behavior, test the running application in a real browser when feasible.
- Prefer Playwright for browser automation.
- Use a visible browser session when requested so I can watch the interactions.
- Verify the actual UI behavior, not only the source code or unit tests.
- Check browser console errors and failed network requests when relevant.
- Report what was tested, what passed, what failed, and what could not be verified.
- Do not claim that browser testing succeeded unless it was actually performed.

### Safety and permissions
- Explain commands that modify the environment, database, or system configuration before running them.
- Ask for approval before destructive or irreversible actions.
- Never expose credentials or overwrite local configuration files containing secrets.


# Repository guidance

## Project purpose and user needs

- **Purpose:** This project provides shoreline-monitoring functionality through OGC Web Processing Service (WPS) endpoints implemented with PyWPS.
- **Users:** Users and client applications submit shoreline-analysis requests and consume the resulting data, statistics, and visualizations.
- **User needs:** Processes should return reliable, understandable results, and generated visualizations should be accessible through valid URLs.
- **Compatibility:** Treat existing WPS interfaces as contracts. Preserve expected process identifiers, input and output identifiers, data types, MIME formats, response structures, and client-facing URLs unless a breaking change is explicitly requested.
- **End-to-end perspective:** Consider the complete flow from client request to WPS process, processing handler, database query, generated result, and client display. A change that works in one component can still break the overall user experience.

## Domain concepts

- A **profile** represents a shoreline profile identified by `gctr.index`. In the WPS interface, `profileid` refers to this value.
- A **transect** is a spatial feature used to analyze shoreline changes over time. Do not confuse a transect's `transect_id` with a profile ID.
- **Profile analysis** and **neighborhood statistics** are separate operations with different spatial scopes and outputs. Do not assume they use identical queries or calculations.
- Spatial calculations use the project's established coordinate reference systems and database conventions.
- The `gctr.geom_3857` geometry is stored in EPSG:3857 and is used for spatial-distance queries expressed in meters. For example, a 5,000 m search radius represents a 5 km neighborhood.
- Generated visualization files are part of the application output contract. Their names, publication paths, accessibility, and returned URLs must remain consistent with the client expectations.

## Architecture

### Backend: PyWPS service

- `pywps.wsgi` creates the Flask application, registers the PyWPS processes, initializes the `/feedback` endpoint, and serves generated files from `data/` through `/data/`.
- The registered WPS processes include `WpsShorelineGetprofile` and `WpsShorelineGetStats`, implemented in `processes/wps_shoreline_getprofile.py` and `processes/wps_shoreline_getstats.py`, respectively.
- WPS process adapters define the client-facing interface, parse inputs, invoke processing logic, and serialize outputs. Keep domain logic and visualization generation out of Flask routes and WPS adapters where they belong in dedicated modules.
- Shoreline handlers query the PostGIS `gctr` and `shorelinemonitor_series` tables and use `generate_dynamic_plot.py` to build interactive Panel/HoloViews visualizations.
- The profile process uses `gctr.index` as the profile ID and finds nearby transects. The statistics process summarizes a 5 km neighborhood.
- `processes/feedback.py` persists feedback to `public.shoreline_feedback`. Its table definition is in `scripts/05_create_feedback_table.sql`.
- The `scripts/` directory contains database-preparation and data-ingestion utilities, including Azure Parquet-to-PostGIS ingestion.
- Profile plots use the `plot_` filename prefix and statistics plots use `stat_`. Generated results are written to `data/` and published through the configured service URL.
- Verify all file paths, configuration keys, and module names against the current code before changing or relying on them. Do not assume that a filename or implementation mentioned in these instructions is still accurate if the repository has changed.

### Frontend: separate codebase

- **Local directory:**  `c:\develop\shoreline-monitor-viewer\`
- **GitHub repository:** `https://github.com/openearth/shoreline-monitor-viewer`
- The frontend is a separate codebase from this WPS backend. Do not assume its source files are present in the current workspace.
- When investigating frontend behaviour, first identify which repository and local directory contain the relevant code. Inspect the frontend code there when it is accessible.
- When working on the integration between the frontend and backend, trace the request and response contracts across both codebases. Check how the frontend constructs WPS requests, identifies profiles and transects, consumes response values, and loads generated visualization URLs.
- For interface bugs, distinguish between problems in the frontend container or layout, the HTML or content returned by the WPS service, and the integration between them. Do not assume that an issue belongs to one side before investigating the evidence.
- If the frontend repository or its runtime is unavailable, state that limitation explicitly. Do not claim to have inspected or tested frontend code that was not accessible.

## Development principles

- Treat existing WPS interfaces as contracts. Avoid breaking changes unless explicitly requested.
- Keep request handling, domain logic, database access, and visualization generation appropriately separated.
- Consider database performance, spatial correctness, generated-file availability, URL construction, and compatibility with existing clients when proposing changes.
- Preserve existing database table names, column names, geometry conventions, and configuration conventions unless there is a justified reason to change them.
- Avoid unrelated refactoring and unnecessary changes.
- Before implementing a significant feature, explain the user problem, the proposed solution, the affected files, relevant risks, and the verification plan.
- When investigating a bug, establish the root cause before recommending or implementing a fix. Trace the relevant execution path and distinguish confirmed findings from assumptions.
- For changes that cross the frontend/backend boundary, explain which side is responsible for each change and how the integration will be verified.

## Build, run, and checks

- Create the application environment with `conda env create -f environment.yml` using Python 3.12, then activate it with `conda activate shoreline_monitor_wps`.
- Copy `configuration.txt.example` to `configuration.txt` and configure its `[database]` PostgreSQL settings before running database-backed processes.
- Start the local Flask/PyWPS service from the repository root with `python pywps.wsgi`.
- Check the current repository for available tests, build commands, and lint configuration before running checks. If no test or lint runner is configured, report that limitation rather than implying that checks exist.
- Do not assume the backend is working solely because the service starts. When feasible, verify the relevant WPS request, database interaction, generated output, and returned URL.
- When a change affects the frontend, test the running application in a real browser when feasible and verify the affected user interaction end to end.

## Repository conventions

- To expose a WPS process, implement a `pywps.app.Process` adapter and register an instance in the `processes` list in `pywps.wsgi`.
- Keep the WPS `identifier`, input/output identifiers, and supported MIME formats aligned with existing clients and relevant XML examples.
- WPS shoreline inputs use complex `application/json` values. Adapters decode the input, call the appropriate handler, and serialize the result into the declared output.
- `profileid` means the `gctr.index` value, not `transect_id`. Queries and client-facing process contracts depend on this distinction.
- Database-backed application settings live in the untracked root `configuration.txt`, using the `[database]` keys shown in `configuration.txt.example`.
- Azure ingestion scripts read storage settings from environment variables; consult `.env.example` for the expected configuration.
- Spatial neighborhood queries use `gctr.geom_3857` in EPSG:3857, where distances such as the 5,000 m radius are expressed in meters.
- Generated profile plots use the `plot_` filename prefix, while statistics plots use `stat_`. Both are published from `data/` and linked through the configured service URL.
- When changing an existing process, inspect its implementation, callers, configuration, and relevant client expectations before changing its behaviour.

## Copilot workflow and response preferences

### Planning and implementation

- For significant changes or requests involving a plan, explain the proposed approach before implementation.
- Identify the files and components expected to change and describe the purpose of each change.
- Highlight relevant risks, dependencies, and testing requirements.
- Do not implement changes when I explicitly ask for investigation or analysis only.
- Avoid unrelated refactoring and unnecessary changes.

### Debugging and investigation

- Start by reproducing or tracing the reported problem when feasible.
- Follow the relevant execution path through the request, processing logic, database interaction, generated output, and frontend consumption as applicable.
- Inspect the actual code and runtime evidence before selecting a root cause.
- For frontend or integration bugs, inspect browser console errors, failed network requests, HTTP responses, and the relevant DOM or layout when feasible.
- Prefer the smallest change that addresses the confirmed root cause.
- Clearly distinguish what is confirmed, what is suspected, and what still needs testing.

### Code explanations

- When explaining how something works, include relevant code snippets from the codebase.
- Trace the execution flow step by step, showing how relevant functions, modules, and components interact.
- Explain technical concepts in plain language before introducing jargon.
- Use concise tables when comparing components, responsibilities, or alternatives.

### Frontend and browser testing

- When changing frontend code or UI behaviour, test the running application in a real browser when feasible.
- Prefer Playwright for browser automation.
- Use a visible browser session when requested so I can watch the interactions.
- Verify actual UI behaviour, not only the source code or unit tests.
- Check browser console errors and failed network requests when relevant.
- Report what was tested, what passed, what failed, and what could not be verified.
- Do not claim that browser testing succeeded unless it was actually performed.

### Safety and permissions

- Explain commands that modify the environment, database, or system configuration before running them.
- Ask for approval before destructive or irreversible actions.
- Never expose credentials or overwrite local configuration files containing secrets.
- Do not execute destructive database operations or overwrite important generated or user data without appropriate approval.