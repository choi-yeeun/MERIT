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

// ---- backbone tabs (EgoLifeQA chart) ----
window.bbTab = function (btn, key) {
  const chart = btn.closest(".chart");
  if (!chart) return;
  chart.querySelectorAll(".bb-tab").forEach((t) => t.classList.toggle("is-active", t === btn));
  chart.querySelectorAll(".bb-panel").forEach((p) => {
    p.hidden = p.dataset.bb !== key;
  });
};

// ---- neighbor-filtering expand/collapse (staged reveal) ----
window.nfExpand = function (btn) {
  const slide = btn.closest(".nfc-slide");
  if (!slide) return;
  const expanded = slide.classList.toggle("nf-expanded");
  btn.textContent = expanded ? "− Collapse" : "＋ Expand ±Δ neighbors";
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
