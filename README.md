# MERIT — Project Page

Static project page for **"Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding"** (ECCV 2026).

Pure HTML/CSS/JS — no build step. Light/dark theme toggle, responsive, figure assets baked in.

## Deploy to `https://choi-yeeun.github.io/MERIT/`

1. Create a **new GitHub repo named exactly `MERIT`** under the `choi-yeeun` account.
2. Push the contents of this folder to the repo root:

   ```bash
   cd MERIT-page
   git init
   git add .
   git commit -m "MERIT project page"
   git branch -M main
   git remote add origin https://github.com/choi-yeeun/MERIT.git
   git push -u origin main
   ```

3. In the repo, go to **Settings → Pages → Build and deployment**, set **Source = Deploy from a branch**, **Branch = `main` / `(root)`**, save.
4. Wait ~1 min, then open **https://choi-yeeun.github.io/MERIT/**.

> The repo name becomes the URL path. All asset paths in `index.html` are **relative**, so the site works correctly under the `/MERIT/` subpath. `.nojekyll` is included so GitHub serves `static/` untouched.

## Editing

- Text & sections: `index.html`
- Colors & theme: `static/css/style.css` (CSS variables at the top; light + `[data-theme="dark"]`)
- Behavior (theme toggle, copy, scroll reveal): `static/js/main.js`
- Figures: `static/images/` (regenerate from the paper PDFs at 3× with PyMuPDF if updated)

## TODO before going live

Buttons (Paper / arXiv / Code) are placeholders linking to `#`. Replace the `href="#"` values in `index.html` once URLs exist, and update the BibTeX entry with final proceedings info.
