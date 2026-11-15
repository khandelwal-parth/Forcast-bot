/* Quant Bot shared helpers: theme toggle, count-up, split-flap text. */
(function () {
    const reduce = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const ease = (t) => 1 - Math.pow(1 - t, 3);

    /* Animate a number into an element. fmt(value) -> string. */
    function countTo(el, from, to, fmt, dur) {
        dur = dur || 800;
        if (reduce() || from === to) { el.textContent = fmt(to); return; }
        const t0 = performance.now();
        (function step(now) {
            const p = Math.min(1, (now - t0) / dur);
            el.textContent = fmt(from + (to - from) * ease(p));
            if (p < 1) requestAnimationFrame(step);
        })(t0);
    }

    /* Departure-board style flip: each letter cycles, then settles left to right. */
    const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    function flip(el, finalText, dur) {
        dur = dur || 700;
        if (reduce()) { el.textContent = finalText; return; }
        const n = finalText.length, t0 = performance.now();
        const settle = (i) => 260 + (dur - 260) * (i / Math.max(1, n - 1));
        (function step(now) {
            const t = now - t0;
            let out = '', done = true;
            for (let i = 0; i < n; i++) {
                const ch = finalText[i];
                if (ch === ' ' || t >= settle(i)) { out += ch; }
                else { out += LETTERS[Math.floor(Math.random() * 26)]; done = false; }
            }
            el.textContent = out;
            if (!done) setTimeout(() => requestAnimationFrame(step), 45);
        })(t0);
    }

    function initTheme() {
        const btn = document.getElementById('themeBtn');
        if (!btn) return;
        const root = document.documentElement;
        const sync = () => btn.setAttribute('aria-label', root.dataset.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
        sync();
        btn.addEventListener('click', () => {
            const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
            root.dataset.theme = next;
            try { localStorage.setItem('qb-theme', next); } catch (e) { /* private mode: fine */ }
            sync();
        });
    }

    /* Add .in-view the first time an element scrolls into view (bars grow, counters run). */
    function onView(selector, cb) {
        const els = document.querySelectorAll(selector);
        if (!('IntersectionObserver' in window)) { els.forEach((e) => { e.classList.add('in-view'); cb && cb(e); }); return; }
        const io = new IntersectionObserver((entries) => {
            entries.forEach((en) => {
                if (!en.isIntersecting) return;
                en.target.classList.add('in-view');
                cb && cb(en.target);
                io.unobserve(en.target);
            });
        }, { threshold: 0.25 });
        els.forEach((e) => io.observe(e));
    }

    window.QB = { reduce, ease, countTo, flip, onView };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initTheme); else initTheme();
})();
