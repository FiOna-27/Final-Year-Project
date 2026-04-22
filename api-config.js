//    Piano Coach · Shared API Config                                        
// All HTML files include this script. It exposes a single global: getApiUrl()
//
// Since app.py now serves the HTML files AND the API, the API is always
// on the same origin as the page. No separate URL needed.
// The stored URL in localStorage is kept as an override for edge cases.

function getApiUrl() {
  const stored = localStorage.getItem('pc_api_base');
  if (stored && stored.trim()) {
    return stored.trim().replace(/\/+$/, '') + '/api';
  }
  // Default: same origin as the current page (works for both localhost and ngrok)
  return window.location.origin + '/api';
}

// Register the service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./service-worker.js')
      .catch(err => console.warn('SW registration failed:', err));
  });
}
