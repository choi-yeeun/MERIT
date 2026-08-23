// ---- theme toggle (persisted) ----
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("merit-theme");
  if (saved) root.setAttribute("data-theme", saved);
  else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
    root.setAttribute("data-theme", "dark");

  window.toggleTheme = function () {
    const cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("merit-theme", next);
  };
})();

// ---- copy bibtex ----
function copyBibtex(btn) {
  const code = document.getElementById("bibtex-code").innerText;
  navigator.clipboard.writeText(code).then(() => {
    const orig = btn.querySelector("span").innerText;
    btn.querySelector("span").innerText = "Copied!";
    setTimeout(() => (btn.querySelector("span").innerText = orig), 1600);
  });
}

// ---- reveal on scroll ----
document.addEventListener("DOMContentLoaded", () => {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  // charts draw in when scrolled into view
  const cio = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in-view");
          // vertical bar chart: reveal the value labels after bars finish growing
          if (e.target.classList.contains("vbar-chart")) {
            const active = e.target.querySelector(".bb-panel:not([hidden])");
            if (active) setTimeout(() => active.classList.add("vals-in"), 550);
          }
          cio.unobserve(e.target);
        }
      });
    },
    { threshold: 0.2 }
  );
  document.querySelectorAll(".chart").forEach((el) => cio.observe(el));
});

// ---- caption expand/collapse ----
window.capToggle = function (btn) {
  const cap = btn.closest(".kvc-cap");
  if (!cap) return;
  const s = cap.querySelector(".cap-short");
  const f = cap.querySelector(".cap-full");
  const showFull = f.hidden;
  f.hidden = !showFull;
  s.hidden = showFull;
  btn.textContent = showFull ? "Show less" : "Show full";
};

// ---- backbone tabs (EgoLifeQA chart) — re-animate bars on switch ----
window.bbTab = function (btn, key) {
  const chart = btn.closest(".chart");
  if (!chart) return;
  chart.querySelectorAll(".bb-tab").forEach((t) => t.classList.toggle("is-active", t === btn));
  chart.querySelectorAll(".bb-panel").forEach((p) => {
    p.hidden = p.dataset.bb !== key;
  });
  // replay the grow-up animation for the now-visible panel, then show values
  chart.querySelectorAll(".bb-panel").forEach((p) => p.classList.remove("vals-in"));
  const active = chart.querySelector('.bb-panel[data-bb="' + key + '"]');
  if (active) {
    active.querySelectorAll(".vb-col").forEach((c) => {
      c.style.transition = "none";
      c.style.height = "0";
      void c.offsetHeight; // force reflow, then restore so it animates up
      c.style.transition = "";
      c.style.height = "";
    });
    setTimeout(() => active.classList.add("vals-in"), 550);
  }
};

// ---- radar: click an axis to spotlight that vertex + print values on the graph ----
document.addEventListener("DOMContentLoaded", () => {
  const SVGNS = "http://www.w3.org/2000/svg";
  const CX = 170, CY = 158, R = 112, MAX = 80;
  const SERIES = [
    { k: "merit", v: [67.2, 70.6, 73.8, 74.4, 71.4], r0: "3.4" },
    { k: "worldmm", v: [62.4, 64.3, 75.4, 62.4, 71.4], r0: "3" },
    { k: "hier", v: [40.0, 56.3, 62.3, 54.4, 52.4], r0: "3" },
    { k: "uniform", v: [47.2, 42.1, 47.5, 53.6, 55.6], r0: "3" },
  ];
  const ang = (i) => (-90 + 72 * i) * Math.PI / 180;
  const pt = (i, v) => [CX + (v / MAX * R) * Math.cos(ang(i)), CY + (v / MAX * R) * Math.sin(ang(i))];
  const perp = (i) => [-Math.sin(ang(i)), Math.cos(ang(i))];
  document.querySelectorAll(".radar-chart").forEach((chart) => {
    const svg = chart.querySelector(".radar");
    const labels = svg.querySelectorAll(".rg-alabel");
    const axes = svg.querySelectorAll(".rg-axis");
    function clear() {
      labels.forEach((l) => l.classList.remove("rg-alabel-active"));
      axes.forEach((a) => a.classList.remove("rg-axis-active"));
      svg.querySelectorAll(".rg-vtxt").forEach((t) => t.remove());
      SERIES.forEach((s) => svg.querySelectorAll("." + s.k + "-dot").forEach((d) => d.setAttribute("r", s.r0)));
    }
    function show(i) {
      clear();
      labels[i].classList.add("rg-alabel-active");
      if (axes[i]) axes[i].classList.add("rg-axis-active");
      const [px, py] = perp(i);
      const max = Math.max.apply(null, SERIES.map((s) => s.v[i]));
      const order = SERIES.map((s, si) => ({ si, val: s.v[i] })).sort((a, b) => b.val - a.val);
      order.forEach((o, rank) => {
        const s = SERIES[o.si];
        const dots = svg.querySelectorAll("." + s.k + "-dot");
        if (dots[i]) dots[i].setAttribute("r", "5");
        const [x, y] = pt(i, s.v[i]);
        const side = rank % 2 === 0 ? 1 : -1;
        const off = 11 + Math.floor(rank / 2) * 3;
        const t = document.createElementNS(SVGNS, "text");
        t.setAttribute("x", (x + px * off * side).toFixed(1));
        t.setAttribute("y", (y + py * off * side).toFixed(1));
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("dominant-baseline", "middle");
        t.setAttribute("class", "rg-vtxt rg-" + s.k + "-vtxt" + (s.v[i] === max ? " rg-best" : ""));
        t.textContent = s.v[i].toFixed(1);
        svg.appendChild(t);
      });
    }
    labels.forEach((l, i) => {
      l.addEventListener("mouseenter", () => show(i));
      l.addEventListener("mouseleave", clear);
    });
  });
});

// ---- custom video player (mint progress bar) ----
document.addEventListener("DOMContentLoaded", () => {
  const fmt = (s) => {
    if (!isFinite(s)) return "0:00";
    const m = Math.floor(s / 60), ss = Math.floor(s % 60);
    return m + ":" + (ss < 10 ? "0" : "") + ss;
  };
  document.querySelectorAll(".vplayer").forEach((p) => {
    const v = p.querySelector("video");
    const fill = p.querySelector(".vp-prog-fill");
    const prog = p.querySelector(".vp-prog");
    const time = p.querySelector(".vp-time");
    if (!v) return;

    const toggle = () => { v.paused ? v.play() : v.pause(); };
    p.querySelectorAll(".vp-big, .vp-toggle").forEach((b) =>
      b.addEventListener("click", (e) => { e.stopPropagation(); toggle(); })
    );
    v.addEventListener("play", () => p.classList.remove("paused"));
    v.addEventListener("pause", () => p.classList.add("paused"));
    p.classList.toggle("paused", v.paused);

    const speedWrap = p.querySelector(".vp-speed");
    if (speedWrap) {
      const sbtn = speedWrap.querySelector(".vp-speed-btn");
      const menu = speedWrap.querySelector(".vp-speed-menu");
      const close = () => {
        speedWrap.classList.remove("open");
        menu.hidden = true;
        sbtn.setAttribute("aria-expanded", "false");
      };
      sbtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = !speedWrap.classList.contains("open");
        speedWrap.classList.toggle("open", open);
        menu.hidden = !open;
        sbtn.setAttribute("aria-expanded", open ? "true" : "false");
      });
      menu.querySelectorAll("li").forEach((li) => {
        li.addEventListener("click", (e) => {
          e.stopPropagation();
          v.playbackRate = parseFloat(li.dataset.rate);
          sbtn.textContent = li.textContent;
          menu.querySelectorAll("li").forEach((x) => x.classList.toggle("active", x === li));
          close();
        });
      });
      document.addEventListener("click", () => { if (speedWrap.classList.contains("open")) close(); });
    }

    v.addEventListener("timeupdate", () => {
      const d = v.duration || 0;
      if (d) fill.style.width = (v.currentTime / d) * 100 + "%";
      time.textContent = fmt(v.currentTime) + " / " + fmt(d);
    });
    v.addEventListener("loadedmetadata", () => {
      time.textContent = "0:00 / " + fmt(v.duration);
    });

    // click-to-seek + drag scrub (mouse & touch via pointer events)
    let dragging = false;
    const seekTo = (clientX) => {
      const r = prog.getBoundingClientRect();
      const x = (clientX - r.left) / r.width;
      if (v.duration) v.currentTime = Math.max(0, Math.min(1, x)) * v.duration;
    };
    prog.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      dragging = true;
      try { prog.setPointerCapture(e.pointerId); } catch (_) {}
      seekTo(e.clientX);
    });
    prog.addEventListener("pointermove", (e) => { if (dragging) seekTo(e.clientX); });
    const endDrag = (e) => {
      if (!dragging) return;
      dragging = false;
      try { prog.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    prog.addEventListener("pointerup", endDrag);
    prog.addEventListener("pointercancel", endDrag);
  });
});

// ---- neighbor-filtering expand/collapse (staged reveal) ----
window.nfExpand = function (btn) {
  const slide = btn.closest(".nfc-slide");
  if (!slide) return;
  const expanded = slide.classList.toggle("nf-expanded");
  btn.textContent = expanded ? "− Collapse" : "＋ Expand Neighbor Clips";
  if (expanded) {
    slide.querySelectorAll("video").forEach((v) => {
      const p = v.play();
      if (p && p.catch) p.catch(() => {});
    });
  }
};

// ---- carousels (multi-instance: multi-key + neighbor filtering) ----
(function () {
  const state = new Map(); // carousel element -> current index
  function render(car) {
    const track = car.querySelector(".kvc-track");
    if (!track) return;
    const idx = state.get(car) || 0;
    track.style.transform = `translateX(-${idx * 100}%)`;
    car.querySelectorAll(".kvc-dot").forEach((d, i) =>
      d.classList.toggle("active", i === idx)
    );
    const slides = track.children;
    for (let i = 0; i < slides.length; i++) {
      slides[i].querySelectorAll("video").forEach((v) => {
        // play only visible videos in the active slide; pause everything else
        if (i === idx && v.offsetParent !== null) {
          const p = v.play();
          if (p && p.catch) p.catch(() => {});
        } else {
          v.pause();
        }
      });
    }
  }
  function goTo(car, i) {
    const n = car.querySelector(".kvc-track").children.length;
    state.set(car, ((i % n) + n) % n);
    // collapse any expanded neighbor-filtering slides when navigating
    car.querySelectorAll(".nfc-slide.nf-expanded").forEach((s) => {
      s.classList.remove("nf-expanded");
      const b = s.querySelector(".nfc-expand");
      if (b) b.textContent = "＋ Expand ±Δ neighbors";
    });
    render(car);
  }
  window.kvcMove = function (btn, dir) {
    const car = btn.closest(".kv-carousel");
    if (car) goTo(car, (state.get(car) || 0) + dir);
  };
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".kv-carousel").forEach((car) => {
      const track = car.querySelector(".kvc-track");
      const dots = car.querySelector(".kvc-dots");
      if (!track || !dots) return;
      state.set(car, 0);
      const n = track.children.length;
      for (let i = 0; i < n; i++) {
        const b = document.createElement("button");
        b.className = "kvc-dot" + (i === 0 ? " active" : "");
        b.setAttribute("aria-label", "Example " + (i + 1));
        b.onclick = () => goTo(car, i);
        dots.appendChild(b);
      }
      render(car);
    });
  });
})();
