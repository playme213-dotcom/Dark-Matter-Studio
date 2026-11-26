# JavaScript Skills for Small Local AI Models

## Purpose
Use this guide to write, explain, debug, and safely modify JavaScript for browsers and Node.js. Prefer clear, predictable code and preserve the project's current behavior and compatibility.

## Workflow
1. Determine the runtime: browser, Node.js, framework, module system, and supported versions.
2. Inspect surrounding code, dependencies, event flow, data shapes, and existing conventions.
3. Identify the expected behavior, constraints, and failure cases.
4. Make a small, targeted change. Preserve existing public interfaces and UI behavior.
5. Run available linting, syntax checks, and focused tests.
6. Report the exact changes and distinguish completed tests from untested assumptions.

## Language fundamentals
- Prefer `const`; use `let` when reassignment is necessary. Avoid `var` in new code.
- Use strict equality (`===`, `!==`) unless coercion is intentional.
- Prefer template literals for interpolation.
- Use descriptive `camelCase` names for variables and functions; use `PascalCase` for classes.
- Prefer `===` comparisons and explicit conditions over confusing truthiness tricks.
- Use optional chaining (`?.`) and nullish coalescing (`??`) only when supported by the target runtime.
- Keep functions focused and avoid unnecessary global variables.
- Use `map`, `filter`, and `find` when they make transformations clearer; use loops when they are simpler.
- Do not mutate shared objects unexpectedly. Make ownership and side effects clear.
- Use modules and explicit exports/imports when supported by the project.

## Async code and errors
- Use `async`/`await` for readable asynchronous flows.
- Handle rejected promises intentionally; do not leave unhandled rejections.
- Use `try/catch` around operations where recovery or useful error reporting is possible.
- Set timeouts and cancellation for network requests when appropriate.
- Do not swallow errors silently or show raw internal stack traces to end users.
- Avoid starting asynchronous work without awaiting it when its completion matters.
- Prevent duplicate submissions and stale responses where relevant.

## Browser DOM and events
- Wait until the DOM is ready when script loading order does not guarantee it.
- Prefer `addEventListener` over inline HTML event attributes.
- Query elements defensively and handle missing elements when they are a valid possibility.
- Use `textContent` for plain text. Avoid `innerHTML` for untrusted or user-provided data.
- Use event delegation for many dynamic child elements when it simplifies handling.
- Remove listeners or clean up timers when components are disposed.
- Preserve keyboard behavior and accessible names for interactive controls.
- Do not rely on brittle selectors or page layout details if stable IDs or data attributes exist.
- Avoid modifying browser APIs or page state in diagnostic-only routines unless explicitly required.

## Network and data
- Check `response.ok` when using `fetch`; HTTP error responses do not automatically reject.
- Validate response data before relying on its fields.
- Use `URL` and `URLSearchParams` to build URLs safely.
- Do not concatenate untrusted data into HTML, SQL, shell commands, or executable code.
- Never hard-code API keys, passwords, or private tokens in frontend JavaScript.
- Do not store sensitive credentials in local storage.
- Treat messages from extensions, sockets, servers, and user-controlled inputs as untrusted.
- Use explicit message schemas and validate command type, payload, and origin where relevant.
- Add timeouts, retries, and deduplication only when they are safe and justified.

## State and UI
- Keep the source of truth for each state value clear.
- Update the UI consistently when state changes.
- Avoid duplicate event handlers and duplicate network commands.
- For multi-step operations, represent status explicitly (for example: idle, running, success, error).
- Keep diagnostic/read-only functions separate from functions that click, navigate, upload, delete, or mutate page state.
- Do not run commands implicitly as a side effect of merely checking or displaying state.
- Confirm destructive actions and handle partial failures when appropriate.

## Compatibility and project editing
- Follow the existing module system (CommonJS, ES modules, bundler, or plain browser scripts).
- Do not introduce syntax unsupported by the project's target browser or Node.js version.
- Inspect callers before changing exported function signatures, event names, message formats, or DOM selectors.
- Preserve IDs, CSS classes, storage keys, route names, API contracts, and extension message types unless explicitly requested.
- Do not replace a whole file when a focused patch is sufficient.
- Avoid adding dependencies without a clear need.
- Keep changes limited to requested files and do not remove working features.

## Security
- Avoid `eval`, `new Function`, unsafe dynamic script injection, and unsafe deserialization.
- Do not use `innerHTML` with untrusted content.
- Validate `postMessage` origins and message schemas.
- For browser extensions, apply least privilege and validate sender, origin, and payload.
- Avoid logging secrets, authentication tokens, or sensitive user content.
- Treat external API responses as untrusted.
- Do not perform destructive actions or transmit data without clear authorization.

## Testing checklist
- Run the project's test and lint commands where available.
- Use `node --check file.js` for Node-compatible syntax checks.
- Test expected behavior, missing elements/data, rejected network calls, and repeated clicks where relevant.
- For DOM code, test in the target browser when possible.
- For bug fixes, verify the original failure is addressed and existing behavior still works.
- Never claim a test passed unless it actually ran and passed.

## Minimal examples

### Safe fetch
```javascript
async function loadJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json();
}
```

### Safe DOM update
```javascript
function setStatus(element, message) {
  if (!element) {
    throw new Error("Status element was not found.");
  }

  element.textContent = message;
}
```

### Prevent duplicate action
```javascript
let isSubmitting = false;

async function submitOnce(submitRequest) {
  if (isSubmitting) return;

  isSubmitting = true;
  try {
    await submitRequest();
  } finally {
    isSubmitting = false;
  }
}
```

## Final rule
Write simple, secure, compatible JavaScript. Preserve existing behavior, validate assumptions, test meaningful cases, and report limitations honestly.
