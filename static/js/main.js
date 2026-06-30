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
});
