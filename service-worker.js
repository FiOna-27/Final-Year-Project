// ── Cache version — increment this string any time you want to force a refresh
// of all cached files (e.g. after updating HTML). Clients will pick up the new
// version on next load and drop the old cache automatically.
const CACHE = 'piano-coach-v2';

// Files to cache on install — all pages and shared scripts
const PRECACHE = [
  './login.html',
  './dashboard.html',
  './piano-coach-v3.5.html',
  './analytics.html',
  './analyser.html',
  './sheetmusic.html',
  './settings.html',
  './theory.html',
  './skill-assessment.html',
  './lesson-plan.html',
  './llm-feedback.html',
  './api-config.js',
  './manifest.json',
];

// Install: cache all app files, but use {cache: 'reload'} to bypass the
// browser's HTTP cache so you always get the freshest files from the server
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c =>
      Promise.all(PRECACHE.map(url =>
        fetch(url, { cache: 'reload' })
          .then(res => c.put(url, res))
          .catch(() => {})   // skip files that 404 rather than failing the whole install
      ))
    ).then(() => self.skipWaiting())
  );
});

// Activate: delete old caches
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch strategy:
// - API calls (/api/...) → network only, never cache (live data)
// - Everything else → cache first, fallback to network
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Never cache API requests
  if (url.pathname.startsWith('/api/') || url.hostname.includes('ngrok')) {
    e.respondWith(fetch(e.request).catch(() => new Response('Offline', { status: 503 })));
    return;
  }

  // Cache first for app files
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(res => {
        if (res && res.status === 200 && res.type === 'basic') {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      }).catch(() => caches.match('./dashboard.html'));
    })
  );
});
