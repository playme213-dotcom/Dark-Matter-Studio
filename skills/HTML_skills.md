# HTML Skills for Small Local AI Models

## Purpose
Use this guide to create, explain, debug, and safely edit semantic HTML. Build pages that are accessible, responsive, maintainable, and compatible with the project's existing structure.

## Workflow
1. Identify the page's purpose, audience, required content, and interactions.
2. Inspect existing markup, stylesheets, scripts, framework, and supported browsers before editing.
3. Use semantic elements and native browser behavior where possible.
4. Make the smallest necessary change and preserve existing IDs, classes, data attributes, and hooks used by CSS or JavaScript.
5. Validate the document structure and test keyboard, responsive, and basic browser behavior.
6. Report files changed and what was actually tested.

## Document structure
- Start with `<!doctype html>` and use `<html lang="...">`.
- Include a UTF-8 charset and a responsive viewport meta tag.
- Give each page a meaningful, unique `<title>`.
- Use semantic landmarks such as `<header>`, `<nav>`, `<main>`, `<section>`, and `<footer>` when appropriate.
- Use headings in a logical hierarchy; do not choose heading levels only for visual size.
- Use `<button>` for actions and `<a href="...">` for navigation.
- Use `<label>` connected to each form control with matching `for` and `id`.
- Set appropriate `type` attributes on form buttons and inputs.
- Use lists for grouped items and tables only for tabular data.
- Use `<figure>` and `<figcaption>` when an image needs a caption.

## Accessibility
- Prefer native HTML elements before adding ARIA.
- Every meaningful image needs useful `alt` text; decorative images should use `alt=""`.
- Controls must be usable with a keyboard and have a visible focus indicator.
- Ensure interactive elements have clear accessible names.
- Do not use color alone to communicate status.
- Associate validation errors and instructions with their fields.
- Use ARIA only when native semantics do not express the required role or state. Keep ARIA roles and attributes accurate.
- Avoid clickable `<div>` or `<span>` elements when a button or link is appropriate.
- Preserve zoom support and do not disable browser scaling.

## Forms and input
- Use the correct input type (`email`, `number`, `date`, etc.) when appropriate.
- Use `required`, `min`, `max`, and `maxlength` when they genuinely match the validation rules.
- Validate important values on the server as well as in the browser.
- Never rely on client-side validation as a security boundary.
- Show clear, specific error messages and preserve user-entered values when safe.
- Do not submit sensitive data to an unexpected destination.

## Links, media, and security
- Use descriptive link text instead of vague phrases like “click here”.
- Set image dimensions or aspect ratios where useful to reduce layout shifts.
- Lazy-load below-the-fold images when appropriate; do not lazy-load critical hero images by default.
- Use HTTPS for external resources when available.
- For links opening a new tab, use `rel="noopener noreferrer"` with `target="_blank"` when appropriate.
- Avoid inline event handlers such as `onclick`; attach behavior in JavaScript.
- Never insert untrusted strings as raw HTML. Prefer `textContent` over `innerHTML`.
- Do not put secrets or private credentials in HTML or client-side code.

## Editing an existing page
- Inspect dependent CSS and JavaScript before renaming or removing IDs, classes, names, or `data-*` attributes.
- Preserve form names, endpoint paths, accessibility labels, and framework-specific attributes unless asked to change them.
- Avoid unnecessary full-page rewrites.
- Keep HTML content separate from presentation and behavior where the project structure supports it.
- Do not add dependencies or frameworks unless requested or clearly necessary.

## Validation checklist
- Check for unclosed or incorrectly nested tags.
- Confirm unique IDs and valid label associations.
- Confirm links and form actions point to intended destinations.
- Test keyboard navigation and focus visibility.
- Test at narrow and wide viewport sizes.
- Check the browser console for relevant errors when scripts are involved.
- Use an HTML validator or project lint command if available; do not claim validation if it was not run.

## Minimal example
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Example page</title>
</head>
<body>
  <header>
    <h1>Example page</h1>
  </header>
  <main>
    <section aria-labelledby="intro-heading">
      <h2 id="intro-heading">Introduction</h2>
      <p>Use semantic HTML to structure content.</p>
    </section>
  </main>
</body>
</html>
```

## Final rule
Use semantic, accessible markup; preserve existing integration hooks; and test behavior rather than assuming the page works.
