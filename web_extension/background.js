// Runs bridge HTTP requests in the extension origin, avoiding page-origin CORS restrictions.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "colibri_bridge_fetch") return;
  const base = "http://127.0.0.1:8765/api/extension";
  const options = {method: message.method || "GET", headers:{}};
  if (message.body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(message.body);
  }
  fetch(`${base}/${message.path}`, options)
    .then(async response => {
      const raw = await response.text();
      let data;
      try { data = raw ? JSON.parse(raw) : {}; } catch (_) { data = {output:raw}; }
      if (!response.ok) throw new Error(data.output || `Bridge HTTP ${response.status}`);
      sendResponse({ok:true, data});
    })
    .catch(error => sendResponse({ok:false, error:String(error?.message || error)}));
  return true;
});
