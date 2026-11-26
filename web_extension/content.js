(() => {
  if (window.__colibriWebExtensionInstalled) return;
  window.__colibriWebExtensionInstalled = true;

  const bridge = (method, path, body) => new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({type:"colibri_bridge_fetch", method, path, body}, response => {
      if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
      if (!response?.ok) return reject(new Error(response?.error || "Bridge request failed"));
      resolve(response.data);
    });
  });

  // Determine the provider from the active website. Without this, tick() throws
  // before sending its first heartbeat and Studio never sees an active tab.
  const host = location.hostname.toLowerCase();
  const provider = host.includes("gemini.google.com") ? "gemini"
    : (host.includes("chatgpt.com") || host.includes("chat.openai.com")) ? "chatgpt"
    : host.includes("claude.ai") ? "claude" : "";
  let busy = false;
  if (!provider) return;

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const post = (path, body) => bridge("POST", path, body);
  const get = path => bridge("GET", path);

  function findInput() {
    const selectors = provider === "chatgpt"
      ? ["#prompt-textarea", "div[contenteditable='true'][data-placeholder]"]
      : provider === "gemini"
      ? ["rich-textarea [contenteditable='true']", ".ql-editor", "div[contenteditable='true'][role='textbox']"]
      : ["div[contenteditable='true'].ProseMirror", "div[contenteditable='true'][data-placeholder]", "fieldset div[contenteditable='true']"];
    for (const selector of selectors) {
      const el = [...document.querySelectorAll(selector)].find(x => x.getClientRects().length && !x.closest("[aria-hidden='true']"));
      if (el) return el;
    }
    return null;
  }

  function setInput(el, value) {
    el.focus();
    if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value")?.set;
      if (setter) setter.call(el, value); else el.value = value;
      el.dispatchEvent(new InputEvent("input", {bubbles:true, inputType:"insertText", data:value}));
    } else {
      el.replaceChildren();
      el.textContent = value;
      el.dispatchEvent(new InputEvent("input", {bubbles:true, inputType:"insertText", data:value}));
    }
  }

  function findSendButton() {
    const selectors = provider === "chatgpt"
      ? ["button[data-testid='send-button']", "button[aria-label*='Send']"]
      : provider === "gemini"
      ? ["button[aria-label*='Send']", "button[mattooltip*='Send']", "button.send-button"]
      : ["button[aria-label*='Send']", "button[data-testid='send-button']"];
    for (const selector of selectors) {
      const el = [...document.querySelectorAll(selector)].find(x => x.getClientRects().length && !x.disabled);
      if (el) return el;
    }
    return null;
  }

  function assistantText() {
    const selectors = provider === "chatgpt"
      ? ["[data-message-author-role='assistant']"]
      : provider === "gemini"
      ? ["message-content", ".model-response-text", "model-response"]
      : ["[data-testid='assistant-message']", ".font-claude-message", "div[data-is-streaming]"];
    for (const selector of selectors) {
      const nodes = [...document.querySelectorAll(selector)].filter(x => x.getClientRects().length);
      if (nodes.length) return (nodes[nodes.length - 1].innerText || nodes[nodes.length - 1].textContent || "").trim();
    }
    return "";
  }

  async function submitPrompt(prompt, requestId) {
    const input = findInput();
    if (!input) throw new Error("Could not find the chat input. Make sure the web chat is fully loaded.");
    const before = assistantText();
    setInput(input, prompt);
    await sleep(350);
    const button = findSendButton();
    if (button) button.click();
    else {
      input.dispatchEvent(new KeyboardEvent("keydown", {key:"Enter", code:"Enter", bubbles:true}));
      input.dispatchEvent(new KeyboardEvent("keyup", {key:"Enter", code:"Enter", bubbles:true}));
    }

    const started = Date.now();
    let previous = "", stable = 0, sawChange = false, lastProgress = "";
    while (Date.now() - started < 8 * 60 * 1000) {
      await sleep(250);
      const current = assistantText();
      if (current && current !== before) {
        sawChange = true;
        // Send each newly rendered assistant update to Studio immediately.
        // The backend turns these snapshots into incremental SSE token events.
        if (current !== lastProgress) {
          try {
            await post("progress", {id:requestId, text:current});
            lastProgress = current;
          } catch (_) {}
        }
        if (current === previous) stable++; else stable = 0;
        previous = current;
        if (stable >= 4) return current;
      }
      // If the assistant starts from an empty message and then fills, a nonempty response is required.
      if (sawChange && current && !document.querySelector("button[aria-label*='Stop'],button[data-testid*='stop']") && stable >= 2) return current;
    }
    if (previous) return previous;
    throw new Error("Timed out waiting for the web chat response.");
  }


  async function attachTextFile(filename, content) {
    // Provider upload controls differ, but all three expose a file input after
    // opening their attachment menu. Populate it with a real browser File.
    let fileInput = [...document.querySelectorAll('input[type="file"]')].find(el => {
      const accept = (el.getAttribute("accept") || "").toLowerCase();
      return !accept || accept.includes("text") || accept.includes("*") || accept.includes(".txt");
    });
    if (!fileInput) {
      const attachSelectors = provider === "chatgpt"
        ? ["button[aria-label*='Attach']", "button[aria-label*='Upload']", "button[data-testid*='attach']"]
        : provider === "gemini"
        ? ["button[aria-label*='Add files']", "button[aria-label*='Attach']", "button[aria-label*='Upload']"]
        : ["button[aria-label*='Attach']", "button[aria-label*='Upload']", "button[aria-label*='Add']"];
      const attachButton = attachSelectors.map(sel => [...document.querySelectorAll(sel)].find(el => el.getClientRects().length)).find(Boolean);
      if (attachButton) attachButton.click();
      await sleep(500);
      fileInput = [...document.querySelectorAll('input[type="file"]')].find(el => {
        const accept = (el.getAttribute("accept") || "").toLowerCase();
        return !accept || accept.includes("text") || accept.includes("*") || accept.includes(".txt");
      });
    }
    if (!fileInput) throw new Error("Could not find the provider's file-upload input; cannot attach initialization text file.");
    const file = new File([content], filename || "colibri_initialization.txt", {type:"text/plain;charset=utf-8"});
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event("input", {bubbles:true}));
    fileInput.dispatchEvent(new Event("change", {bubbles:true}));
    await sleep(900);
  }

  async function submitInitializationFile(task) {
    await attachTextFile(task.filename, task.content);
    await submitPrompt(task.prompt, task.id);
  }

  async function tick() {
    if (busy) return;
    try {
      await post("heartbeat", {provider, url:location.href});
      const task = await get(`poll?provider=${provider}`);
      if (task.status !== "prompt" && task.status !== "file") return;
      busy = true;
      try {
        const answer = task.status === "file"
          ? await submitInitializationFile(task)
          : await submitPrompt(task.prompt, task.id);
        await post("result", {id:task.id, text:answer});
      } catch (error) {
        await post("fail", {id:task.id, error:String(error?.message || error)});
      } finally {
        busy = false;
      }
    } catch (_) {
      // Studio may not be running yet. Retry on the next tick.
    }
  }

  tick();
  setInterval(tick, 1200);
})();
